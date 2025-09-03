#!/usr/bin/env python3

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import spacy
import mysql.connector
from paddleocr import PaddleOCR
import math
from collections import Counter
import re
import threading
import json
from pathlib import Path

ctk.set_appearance_mode("light")  
ctk.set_default_color_theme("blue")

def preprocess_image(image):
   
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()
    
    print("Step 1: Converting to grayscale...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print("Step 2: Enhancing contrast...")
    enhanced = cv2.equalizeHist(gray)
    
    print("Step 3: Reducing noise...")
    denoised = cv2.medianBlur(enhanced, 3)  
    
    print("Step 4: Applying threshold...")
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
    
    print("Step 5: Cleaning text...")
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    print("Step 6: Final resize...")
    height, width = cleaned.shape
    if height < 800:
        scale_factor = 800 / height
        new_width = int(width * scale_factor)
        final = cv2.resize(cleaned, (new_width, 800), interpolation=cv2.INTER_CUBIC)
    else:
        final = cleaned
    
    print("Preprocessing completed!")
    # Convert back to PIL Image
    final_pil = Image.fromarray(final)
    
    return final_pil



def create_ngrams(text, n_min=2, n_max=4):
    
    text = text.lower()
    ngrams = []
    
    for n in range(n_min, n_max + 1):
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            if any(c.isalpha() for c in ngram):
                ngrams.append(ngram)
    
    return ngrams

def compute_term_frequency(ngrams):
    
    total_terms = len(ngrams)
    if total_terms == 0:
        return {}
    
    term_counts = Counter(ngrams)
    tf = {}
    
    # TF = (Number of times term appears) / (Total number of terms)
    for term, count in term_counts.items():
        tf[term] = count / total_terms
    
    return tf

def compute_inverse_document_frequency(documents_ngrams):
    
    total_docs = len(documents_ngrams)
    if total_docs == 0:
        return {}
    
    # Get all unique terms
    all_terms = set()
    for ngrams in documents_ngrams:
        all_terms.update(ngrams)
    
    idf = {}
    
    # IDF = log(Total documents / Documents containing term)
    for term in all_terms:
        docs_containing_term = sum(1 for ngrams in documents_ngrams if term in ngrams)
        if docs_containing_term > 0:
            idf[term] = math.log(total_docs / docs_containing_term)
        else:
            idf[term] = 0
    
    return idf

def compute_tfidf_vector(tf, idf, all_terms):
   
    tfidf_vector = []
    
    # TF-IDF = TF * IDF
    for term in all_terms:
        tf_value = tf.get(term, 0)
        idf_value = idf.get(term, 0)
        tfidf_value = tf_value * idf_value
        tfidf_vector.append(tfidf_value)
    
    return tfidf_vector

def compute_cosine_similarity(vector1, vector2):
    
    if len(vector1) != len(vector2):
        return 0
    
    # Dot product
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    
    # Magnitude of vector1
    magnitude1 = math.sqrt(sum(a * a for a in vector1))
    
    # Magnitude of vector2
    magnitude2 = math.sqrt(sum(a * a for a in vector2))
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    # Cosine similarity = dot_product / (magnitude1 * magnitude2)
    cosine_sim = dot_product / (magnitude1 * magnitude2)
    
    return cosine_sim

def custom_tfidf_cosine_similarity(search_term, document_list):
    
    print("🔧 Starting custom TF-IDF and Cosine Similarity computation...")
    
    # Step 1: Prepare all documents (search term + document list)
    all_documents = [search_term] + document_list
    print(f"📄 Processing {len(all_documents)} documents (1 query + {len(document_list)} targets)")
    
    # Step 2: Create n-grams for all documents
    print("🔤 Creating character n-grams...")
    documents_ngrams = []
    for i, doc in enumerate(all_documents):
        ngrams = create_ngrams(doc, n_min=2, n_max=4)
        documents_ngrams.append(ngrams)
        # print(f"   Document {i}: {len(ngrams)} n-grams generated")
    
    # Step 3: Compute IDF for all terms
    print("📊 Computing Inverse Document Frequency (IDF)...")
    idf = compute_inverse_document_frequency(documents_ngrams)
    print(f"   Total unique terms: {len(idf)}")
    
    # Step 4: Get all unique terms for vector space
    all_terms = list(idf.keys())
    print(f"🔢 Vector space dimension: {len(all_terms)}")
    
    # Step 5: Compute TF-IDF vectors for all documents
    print("⚡ Computing TF-IDF vectors...")
    tfidf_vectors = []
    for i, ngrams in enumerate(documents_ngrams):
        tf = compute_term_frequency(ngrams)
        tfidf_vector = compute_tfidf_vector(tf, idf, all_terms)
        tfidf_vectors.append(tfidf_vector)
        # print(f"   Document {i}: TF-IDF vector computed")
    
    # Step 6: Compute cosine similarities
    print("🎯 Computing cosine similarities...")
    query_vector = tfidf_vectors[0]  # First vector is the search term
    similarities = []
    
    for i in range(1, len(tfidf_vectors)):  # Skip the query vector
        similarity = compute_cosine_similarity(query_vector, tfidf_vectors[i])
        similarities.append(similarity)
        # print(f"   Similarity with document {i}: {similarity:.4f}")
    
    print("✅ Custom TF-IDF and Cosine Similarity computation completed!")
    return similarities


class MedicalInfoApp:
    def __init__(self):
        # Initialize variables
        self.cap = None
        self.ocr_model = PaddleOCR(lang='en', use_gpu=False)
        self.current_image = None
        self.original_image = None
        self.preprocessed_image = None
        self.ocr_results = None
        self.ner_results = None
        self.database_results = None
        self.analysis_complete = False
        self.ocr_complete = False
        self.ner_complete = False
        self.db_complete = False
        self.ocr_text = ""
        
        # Load NER model
        try:
            self.nlp_ner = spacy.load("../model-best")
        except Exception as e:
            print(f"Warning: Could not load NER model: {e}")
            self.nlp_ner = None
        
        # Main window setup
        self.window = ctk.CTk()
        self.window.title("🏥 Medical Info Provider")
        self.window.geometry("1600x1000")
        self.window.minsize(1400, 900)
        
        # Center window on screen
        self.center_window()
        
        # Configure fonts
        self.title_font = ctk.CTkFont(family="Inter", size=24, weight="bold")
        self.heading_font = ctk.CTkFont(family="Inter", size=18, weight="bold")
        self.normal_font = ctk.CTkFont(family="Inter", size=14)
        self.small_font = ctk.CTkFont(family="Inter", size=12)
        
        # Consistent button color
        self.button_color = "#2563eb"
        self.button_hover = "#1d4ed8"
        
        self.setup_ui()
        
    def center_window(self):
        """Center the window on screen"""
        self.window.update_idletasks()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - 1600) // 2
        y = (screen_height - 1000) // 2
        self.window.geometry(f"1600x1000+{x}+{y}")
    
    def setup_ui(self):
        """Setup the main user interface"""
        # Configure grid weights
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(1, weight=1)
        
        # Header
        self.create_header()
        
        # Main content area
        self.create_main_content()
        
        # Status bar
        self.create_status_bar()
    
    def create_header(self):
       
        header_frame = ctk.CTkFrame(self.window, height=80, corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_columnconfigure(1, weight=1)
        
        # App title
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w", padx=30, pady=20)
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="🏥 Medical Info Provider",
            font=self.title_font,
            text_color="#2563eb"
        )
        title_label.pack(side="left")
        
        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="AI-Powered Medicine Analysis System",
            font=self.small_font,
            text_color="#64748b"
        )
        subtitle_label.pack(side="left", padx=(20, 0))
        
        # Header controls
        controls_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        controls_frame.grid(row=0, column=2, sticky="e", padx=30, pady=20)
        
        info_btn = ctk.CTkButton(
            controls_frame,
            text="ℹ️ Info",
            width=80,
            height=40,
            font=self.normal_font,
            fg_color=self.button_color,
            hover_color=self.button_hover,
            command=self.show_info
        )
        info_btn.pack(side="right", padx=5)
    
    def create_main_content(self):
        
        content_frame = ctk.CTkFrame(self.window, corner_radius=15)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(10, 20))
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(1, weight=1)
        
        # Upload section
        self.create_upload_section(content_frame)
        
        # Analysis section  
        self.create_analysis_section(content_frame)
    
    def create_upload_section(self, parent):
        
        upload_frame = ctk.CTkFrame(parent, height=120)
        upload_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        upload_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Section title
        upload_title = ctk.CTkLabel(
            upload_frame,
            text="📁 Upload Medicine Image",
            font=self.heading_font,
            text_color="#1e293b"
        )
        upload_title.grid(row=0, column=0, columnspan=2, pady=(20, 20))
        
        subtitle = ctk.CTkLabel(
            upload_frame,
            text="Choose an image file or use live camera feed",
            font=self.small_font,
            text_color="#64748b"
        )
        subtitle.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Upload buttons
        self.upload_btn = ctk.CTkButton(
            upload_frame,
            text="📤 Upload File",
            font=self.normal_font,
            height=50,
            width=200,
            fg_color=self.button_color,
            hover_color=self.button_hover,
            command=lambda: self.process_image("upload")
        )
        self.upload_btn.grid(row=2, column=0, padx=20, pady=(0, 20))
        
        self.livefeed_btn = ctk.CTkButton(
            upload_frame,
            text="📹 Live Feed",
            font=self.normal_font,
            height=50,
            width=200,
            fg_color=self.button_color,
            hover_color=self.button_hover,
            command=self.show_live_feed
        )
        self.livefeed_btn.grid(row=2, column=1, padx=20, pady=(0, 20))
    
    def create_analysis_section(self, parent):
       
        analysis_frame = ctk.CTkFrame(parent)
        analysis_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        analysis_frame.grid_columnconfigure(0, weight=2)
        analysis_frame.grid_columnconfigure(1, weight=1)
        analysis_frame.grid_rowconfigure(0, weight=1)
        
        # Image preview area
        self.image_frame = ctk.CTkFrame(analysis_frame)
        self.image_frame.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
        
        self.image_label = ctk.CTkLabel(
            self.image_frame,
            text="🖼️\n\nNo image loaded\n\nClick 'Upload File' to begin analysis",
            font=self.normal_font,
            text_color="#64748b",
            width=750,
            height=550
        )
        self.image_label.pack(expand=True, fill="both", pady=50, padx=50)
        
        # Control panel
        controls_frame = ctk.CTkFrame(analysis_frame)
        controls_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
        
        controls_title = ctk.CTkLabel(
            controls_frame,
            text="🔧 Analysis Tools",
            font=self.heading_font,
            text_color="#1e293b"
        )
        controls_title.pack(pady=(20, 30))
        
        # Analysis buttons
        self.preprocess_btn = ctk.CTkButton(
            controls_frame,
            text="🔧 Show Preprocessed",
            font=self.normal_font,
            height=40,
            fg_color=self.button_color,
            hover_color=self.button_hover,
            command=self.toggle_preprocessed,
            state="disabled"
        )
        self.preprocess_btn.pack(pady=10, padx=20, fill="x")
        
        self.ocr_boxes_btn = ctk.CTkButton(
            controls_frame,
            text="📊 Show OCR Boxes",
            font=self.normal_font,
            height=40,
            fg_color=self.button_color,
            hover_color=self.button_hover,
            command=self.toggle_ocr_boxes,
            state="disabled"
        )
        self.ocr_boxes_btn.pack(pady=10, padx=20, fill="x")
        
        self.analysis_btn = ctk.CTkButton(
            controls_frame,
            text="📋 OCR/NER Analysis",
            font=self.normal_font,
            height=40,
            fg_color=self.button_color,
            hover_color=self.button_hover,
            command=self.show_analysis_details,
            state="disabled"
        )
        self.analysis_btn.pack(pady=10, padx=20, fill="x")
        
        self.database_btn = ctk.CTkButton(
            controls_frame,
            text="💊 Database Results",
            font=self.normal_font,
            height=40,
            fg_color=self.button_color,
            hover_color=self.button_hover,
            command=self.show_database_results,
            state="disabled"
        )
        self.database_btn.pack(pady=10, padx=20, fill="x")
        
        # Progress bar
        progress_label = ctk.CTkLabel(
            controls_frame,
            text="Analysis Progress:",
            font=self.small_font,
            text_color="#64748b"
        )
        progress_label.pack(pady=(30, 5), padx=20, anchor="w")
        
        self.progress_bar = ctk.CTkProgressBar(controls_frame)
        self.progress_bar.pack(pady=(0, 20), padx=20, fill="x")
        self.progress_bar.set(0)

    
    def create_status_bar(self):
       
        self.status_frame = ctk.CTkFrame(self.window, height=35, corner_radius=0)
        self.status_frame.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready • Medical Info Provider v2.0",
            font=self.small_font,
            text_color="#64748b"
        )
        self.status_label.pack(side="left", padx=20, pady=8)
    
    def update_status(self, message):
        
        self.status_label.configure(text=f"{message} • Medical Info Provider v2.0")
        self.window.update()
    
    def process_image(self, source):
       
        if source == "upload":
            file_types = [
                ('Image files', '*.png *.jpg *.jpeg *.gif *.bmp'),
                ('All files', '*.*')
            ]
            filename = filedialog.askopenfilename(filetypes=file_types)
            if filename:
                self.load_image(filename)
        elif source == "capture" and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.cap.release()
                # Convert OpenCV image to PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                self.analyze_image(pil_image)
    
    def load_image(self, filepath):
        
        try:
            # Get file extension
            file_ext = os.path.splitext(filepath)[1].lower()
            
            # Check if file format is supported
            supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
            
            if file_ext not in supported_formats:
                messagebox.showerror(
                    "Unsupported Format", 
                    f"File format '{file_ext}' is not supported.\n\n"
                    f"Supported formats: {', '.join(sorted(supported_formats))}"
                )
                return
            
            # Check file size (limit to 50MB)
            file_size = os.path.getsize(filepath)
            if file_size > 50 * 1024 * 1024:  # 50MB
                messagebox.showerror(
                    "File Too Large", 
                    f"File size ({file_size / (1024*1024):.1f}MB) exceeds the 50MB limit."
                )
                return
            
            # Try to open the image
            pil_image = Image.open(filepath)
            
            # Check image dimensions
            width, height = pil_image.size
            if width < 50 or height < 50:
                messagebox.showerror(
                    "Image Too Small", 
                    f"Image dimensions ({width}x{height}) are too small.\n"
                    "Minimum size: 50x50 pixels"
                )
                return
            
            if width > 10000 or height > 10000:
                messagebox.showwarning(
                    "Large Image", 
                    f"Image dimensions ({width}x{height}) are very large.\n"
                    "Processing may take longer."
                )
            
            # Convert to RGB if needed (for JPEG compatibility)
            if pil_image.mode in ('RGBA', 'P'):
                pil_image = pil_image.convert('RGB')
            
            self.update_status(f"Loaded image: {os.path.basename(filepath)} ({width}x{height})")
            self.analyze_image(pil_image)
            
        except FileNotFoundError:
            messagebox.showerror("File Not Found", f"Could not find the file:\n{filepath}")
        except PermissionError:
            messagebox.showerror("Permission Denied", f"Permission denied accessing:\n{filepath}")
        except Image.UnidentifiedImageError:
            messagebox.showerror(
                "Invalid Image", 
                f"The file is not a valid image or is corrupted:\n{os.path.basename(filepath)}"
            )
        except Exception as e:
            messagebox.showerror(
                "Error Loading Image", 
                f"An unexpected error occurred:\n{str(e)}\n\nFile: {os.path.basename(filepath)}"
            )
    
    def analyze_image(self, pil_image):
       
        self.original_image = pil_image.copy()
        self.reset_analysis_state()
        
        # Display image without cropping - maintain aspect ratio
        self.display_image(pil_image)
        
        # Enable control buttons
        self.preprocess_btn.configure(state="normal")
        self.ocr_boxes_btn.configure(state="normal")
        self.analysis_btn.configure(state="normal")
        
        # Start analysis in background thread
        self.update_status("Starting analysis...")
        threading.Thread(target=self.run_analysis, daemon=True).start()
    
    def reset_analysis_state(self):
       
        self.ocr_results = None
        self.ner_results = None
        self.database_results = None
        self.analysis_complete = False
        self.ocr_complete = False
        self.ner_complete = False
        self.db_complete = False
        self.ocr_text = ""
        self.database_btn.configure(state="disabled")
    
    def display_image(self, pil_image):
        
        # Get larger display area since we removed status section
        label_width = 700  # Increased from 550
        label_height = 500  # Increased from 400
        
        # Calculate aspect ratio preserving resize
        img_width, img_height = pil_image.size
        aspect_ratio = img_width / img_height
        
        # Calculate the best fit without cropping
        if aspect_ratio > label_width / label_height:
            # Image is wider than label ratio
            new_width = min(label_width, img_width)  # Don't upscale
            new_height = int(new_width / aspect_ratio)
        else:
            # Image is taller than label ratio
            new_height = min(label_height, img_height)  # Don't upscale
            new_width = int(new_height * aspect_ratio)
        
        # Ensure minimum readable size
        if new_width < 200 or new_height < 150:
            if aspect_ratio > 1:
                new_width = 300
                new_height = int(300 / aspect_ratio)
            else:
                new_height = 300
                new_width = int(300 * aspect_ratio)
        
        # Resize image maintaining aspect ratio
        display_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(display_image)
        
        self.image_label.configure(image=photo, text="", width=label_width, height=label_height)
        self.image_label.image = photo
        self.current_image = pil_image
    
    def run_analysis(self):
        
        try:
            # Update progress
            self.progress_bar.set(0.1)
            self.update_status("Preprocessing image...")
            
            # Preprocess image
            self.preprocessed_image = preprocess_image(self.original_image)
            self.progress_bar.set(0.3)
            
            # Save images
            save_folder = Path('../uploaded_images')
            save_folder.mkdir(exist_ok=True)
            
            original_path = save_folder / 'med_original.jpg'
            processed_path = save_folder / 'med.jpg'
            
            self.original_image.save(original_path)
            self.preprocessed_image.save(processed_path)
            
            self.update_status("Running OCR...")
            self.progress_bar.set(0.5)
            
            # Run OCR
            self.ocr_results = self.ocr_model.ocr(str(processed_path))
            self.ocr_complete = True
            self.progress_bar.set(0.7)
            
            # Extract text
            self.ocr_text = self.extract_text_from_ocr(self.ocr_results)
            
            self.update_status("Running NER...")
            
            # Run NER if available
            if self.nlp_ner and self.ocr_text.strip():
                doc = self.nlp_ner(self.ocr_text)
                self.ner_results = [(ent.text, ent.label_) for ent in doc.ents]
                self.ner_complete = True
            else:
                self.ner_results = []
                self.ner_complete = True
            
            self.progress_bar.set(0.85)
            self.update_status("Searching database...")
            
            # Search database
            self.search_database()
            
            self.progress_bar.set(1.0)
            self.analysis_complete = True
            self.update_status("Analysis complete")
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            messagebox.showerror("Analysis Error", f"Failed to analyze image: {str(e)}")
    
    def extract_text_from_ocr(self, ocr_results):
       
        if not ocr_results:
            return ""
        
        text_parts = []
        for result_group in ocr_results:
            if result_group:
                for detection in result_group:
                    text_parts.append(detection[1][0])
        
        return " ".join(text_parts)
    
    def search_database(self):
       
        try:
            # Check if OCR extracted any meaningful text
            if not self.ocr_text or len(self.ocr_text.strip()) < 5:
                self.database_results = self.create_no_result_message("insufficient_text")
                self.database_btn.configure(state="normal")  # Enable button to show error message
                return
            
            # Extract medicine names from NER results
            medicine_names = [text for text, label in self.ner_results 
                            if label in ['COMPOSITION NAME', 'COMMON NAME']]
            
            # If no NER results, try to extract potential medicine names from OCR text
            if not medicine_names:
                # Look for common medicine patterns and keywords
                ocr_words = self.ocr_text.split()
                potential_names = []
                
                # Filter out common non-medicine words
                excluded_words = {
                    'tablet', 'tablets', 'capsule', 'capsules', 'mg', 'ml', 'each', 'contains',
                    'manufactured', 'by', 'for', 'pharma', 'pharmaceuticals', 'ltd', 'limited',
                    'pack', 'strip', 'blister', 'dose', 'dosage', 'strength', 'composition',
                    'generic', 'brand', 'medicine', 'drug', 'rx', 'only', 'prescription'
                }
                
                for word in ocr_words:
                    if (len(word) > 3 and 
                        word.lower() not in excluded_words and 
                        not word.isdigit() and 
                        any(c.isalpha() for c in word)):
                        potential_names.append(word)
                
                medicine_names = potential_names[:3]  # Take first 3 potential names
            
            # If still no medicine names found, it might not be a valid medicine image
            if not medicine_names:
                self.database_results = self.create_no_result_message("no_medicine_detected")
                self.database_btn.configure(state="normal")  # Enable button to show error message
                return
            
            # Connect to database
            conn = mysql.connector.connect(
                host="localhost",
                username="root",
                password="jijash001",
                database="medicine"
            )
            cursor = conn.cursor()
            
            # Search for medicines using cosine similarity
            best_match = None
            best_results = None
            best_score = 0
            
            for name in medicine_names:
                match_results, score = self.find_best_match_in_db(name, cursor)
                if match_results and score > best_score:
                    best_results = match_results
                    best_match = name
                    best_score = score
            
            cursor.close()
            conn.close()
            
            # Evaluate results based on confidence score
            if best_results and best_score > 0.3:  # Minimum confidence threshold
                self.database_results = self.format_database_results(best_results, best_match, best_score)
                self.database_btn.configure(state="normal")
            elif best_results and best_score > 0.1:  # Low confidence
                self.database_results = self.create_low_confidence_message(medicine_names, best_score)
                self.database_btn.configure(state="normal")  # Enable button to show error message
            else:
                self.database_results = self.create_no_result_message("no_match_found", medicine_names)
                self.database_btn.configure(state="normal")  # Enable button to show error message
            
        except Exception as e:
            self.database_results = f"Database connection error: {str(e)}\n\nPlease check your database connection and try again."
            self.database_btn.configure(state="normal")  # Enable button to show error message
    
    def find_best_match_in_db(self, search_term, cursor):
        """Find the best matching drug name using custom TF-IDF and cosine similarity (FROM SCRATCH)"""
        try:
            print(f"🔍 Searching for: '{search_term}' using custom algorithm...")
            
            # Get all drug names
            cursor.execute("SELECT DISTINCT DRUG_NAME FROM med_info")
            all_drugs = cursor.fetchall()
            drug_names = [drug[0].lower() for drug in all_drugs if drug[0]]
            
            if not drug_names:
                print("❌ No drug names found in database")
                return None, 0
            
            print(f"📋 Comparing against {len(drug_names)} drugs in database")
            
            # Use our custom TF-IDF and cosine similarity implementation
            search_term_lower = search_term.lower()
            similarities = custom_tfidf_cosine_similarity(search_term_lower, drug_names)
            
            # Find the best match
            if not similarities:
                print("❌ No similarities computed")
                return None, 0
            
            best_match_idx = similarities.index(max(similarities))
            best_score = similarities[best_match_idx] * 100  # Convert to percentage
            
            print(f"🎯 Best match: '{drug_names[best_match_idx]}' with score: {best_score:.2f}%")
            
            # Set threshold for acceptance
            if best_score >= 37:
                original_drug_names = [drug[0] for drug in all_drugs if drug[0]]
                best_drug_name = original_drug_names[best_match_idx]
                
                print(f"✅ Match accepted: '{best_drug_name}'")
                query = "SELECT * FROM med_info WHERE DRUG_NAME = %s"
                cursor.execute(query, (best_drug_name,))
                results = cursor.fetchall()
                return results, best_score
            else:
                print(f"⚠️ Match rejected: Score {best_score:.2f}% below threshold (60%)")
                return None, best_score
                
        except Exception as e:
            print(f"❌ Error in custom similarity matching: {e}")
            return None, 0
    
    def format_database_results(self, results, drug_name, similarity_score):
        """Format database results for display"""
        if not results:
            return "No results found."
        
        column_labels = [
            "DRUG NAME", "DOSAGE FORM AND STRENGTH", "INDICATIONS",
            "CONTRAINDICATIONS OR PRECAUTIONS", "DOSAGE SCHEDULE", 
            "ADVERSE EFFECTS", "DRUG AND FOOD INTERACTIONS"
        ]
        
        content = f"🔍 Search Results for: {drug_name}\n"
        content += f"📊 Similarity Score: {similarity_score:.1f}%\n"
        content += "=" * 80 + "\n\n"
        
        for row in results:
            for i, value in enumerate(row):
                if value and i < len(column_labels):
                    content += f"📌 {column_labels[i]}:\n{value}\n\n"
        
        return content
    
    def create_no_result_message(self, error_type, medicine_names=None):
        """Create comprehensive no-result messages based on error type"""
        
        if error_type == "insufficient_text":
            return """❌ Sorry, couldn't provide the result.

🔍 ANALYSIS SUMMARY:
• OCR detected insufficient or unclear text
• Unable to extract meaningful medicine information

⚠️ CHECK THE FOLLOWING:

1. 📸 IMAGE CLARITY:
   • Image may not be clear enough
   • Try taking a well-lit, focused photo
   • Ensure text is clearly visible and readable
   • Avoid blurry or pixelated images

2. 📦 PACKAGING TYPE:
   • Image may not be of a blister packet
   • Our system works best with blister pack labels
   • Other packaging types are not currently supported:
     * Bottles, jars, or containers
     * Outer boxes or cartons
     * Liquid medicine bottles
     * Injection vials or ampoules

💡 SUGGESTIONS:
• Take a photo of the blister pack directly
• Ensure good lighting without shadows
• Hold the camera steady for sharp focus
• Make sure the medicine name is clearly visible"""

        elif error_type == "no_medicine_detected":
            return """❌ Sorry, couldn't provide the result.

🔍 ANALYSIS SUMMARY:
• No medicine entities detected in the image
• The image may not contain recognizable medicine information

⚠️ CHECK THE FOLLOWING:

1. 📸 IMAGE CLARITY:
   • Image may not be clear enough for text recognition
   • Medicine name/composition text may be too small or blurred
   • Try taking a closer, well-focused photo

2. 📦 PACKAGING TYPE:
   • Image may not be of a blister packet
   • Please ensure you're photographing:
     ✅ Medicine blister packs with printed text
     ✅ Clear medicine labels with drug information
   • NOT supported:
     ❌ Empty packages or boxes
     ❌ Bottles without clear labels
     ❌ Damaged or torn packages

💡 SUGGESTIONS:
• Take a direct photo of the medicine blister pack
• Ensure the medicine name is clearly visible
• Use good lighting and avoid reflections
• Make sure the text is not obscured or damaged"""

        elif error_type == "no_match_found":
            detected_text = ', '.join(medicine_names) if medicine_names else "None"
            return f"""❌ Sorry, couldn't provide the result.

🔍 ANALYSIS SUMMARY:
• Detected text: {detected_text}
• No matching medicine found in our database

⚠️ CHECK THE FOLLOWING:

1. 📸 IMAGE CLARITY:
   • Medicine name may not be clearly readable
   • OCR might have misread the text
   • Try taking a clearer, more focused photo

2. 📦 PACKAGING TYPE:
   • Image may not be of a supported blister packet
   • Some international or generic medicines may not be in our database
   • Ensure you're photographing standard pharmaceutical blister packs

3. 🗄️ DATABASE COVERAGE:
   • This medicine might not be in our current database
   • Our database primarily contains commonly prescribed medicines
   • Regional or specialty medicines may not be available

💡 SUGGESTIONS:
• Try a different angle or lighting
• Ensure the complete medicine name is visible
• Check if it's a standard pharmaceutical blister pack
• Consult a healthcare professional for unknown medicines"""

        else:
            return """❌ Sorry, couldn't provide the result.

⚠️ CHECK THE FOLLOWING:

1. 📸 IMAGE CLARITY:
   • Image may not be clear enough

2. 📦 PACKAGING TYPE:
   • Image may not be of a blister packet
   • Other packaging types are not accepted

💡 Please try again with a clear photo of a medicine blister pack."""

    def create_low_confidence_message(self, medicine_names, confidence_score):
        """Create message for low confidence matches"""
        detected_text = ', '.join(medicine_names) if medicine_names else "None"
        return f"""⚠️ Low confidence result detected.

🔍 ANALYSIS SUMMARY:
• Detected text: {detected_text}
• Confidence score: {confidence_score:.1f}%
• Results may not be accurate

❌ Sorry, couldn't provide a reliable result.

⚠️ CHECK THE FOLLOWING:

1. 📸 IMAGE CLARITY:
   • Image may not be clear enough for accurate recognition
   • Medicine text appears partially readable but unclear
   • Try taking a sharper, well-lit photo

2. 📦 PACKAGING TYPE:
   • Image may not be of a standard blister packet
   • Text might be partially obscured or damaged
   • Ensure the complete medicine label is visible

💡 SUGGESTIONS:
• Retake the photo with better lighting
• Ensure the medicine name is completely visible
• Use a steady hand to avoid blur
• Try a different angle if text is partially hidden"""
    
    def toggle_preprocessed(self):
        """Toggle between original and preprocessed image"""
        if hasattr(self, 'showing_preprocessed') and self.showing_preprocessed:
            self.display_image(self.original_image)
            self.preprocess_btn.configure(text="🔧 Show Preprocessed")
            self.showing_preprocessed = False
        else:
            if self.preprocessed_image:
                self.display_image(self.preprocessed_image)
                self.preprocess_btn.configure(text="🔧 Show Original")
                self.showing_preprocessed = True
    
    def toggle_ocr_boxes(self):
        """Toggle OCR bounding boxes"""
        if not self.ocr_results:
            return
        
        if hasattr(self, 'showing_boxes') and self.showing_boxes:
            # Hide boxes
            base_image = self.preprocessed_image if hasattr(self, 'showing_preprocessed') and self.showing_preprocessed else self.original_image
            self.display_image(base_image)
            self.ocr_boxes_btn.configure(text="📊 Show OCR Boxes")
            self.showing_boxes = False
        else:
            # Show boxes
            base_image = self.preprocessed_image if hasattr(self, 'showing_preprocessed') and self.showing_preprocessed else self.original_image
            image_with_boxes = self.draw_ocr_boxes(base_image, self.ocr_results)
            self.display_image(image_with_boxes)
            self.ocr_boxes_btn.configure(text="📊 Hide OCR Boxes")
            self.showing_boxes = True
    
    def draw_ocr_boxes(self, pil_image, ocr_results):
        """Draw OCR bounding boxes on image"""
        if not ocr_results:
            return pil_image
        
        img_with_boxes = pil_image.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
        color_index = 0
        
        for result_group in ocr_results:
            if result_group:
                for detection in result_group:
                    bbox = detection[0]
                    text = detection[1][0]
                    confidence = detection[1][1]
                    
                    # Convert bbox to proper format
                    points = [(int(point[0]), int(point[1])) for point in bbox]
                    
                    # Draw bounding box
                    color = colors[color_index % len(colors)]
                    draw.polygon(points, outline=color, width=3)
                    
                    # Add text label
                    label = f"{text} ({confidence:.2f})"
                    text_position = (points[0][0], max(0, points[0][1] - 20))
                    draw.text(text_position, label, fill=color)
                    
                    color_index += 1
        
        return img_with_boxes
    
    def show_analysis_details(self):
        """Show detailed OCR/NER analysis"""
        if not self.ocr_results and not self.ner_results:
            messagebox.showinfo("Analysis", "No analysis results available.")
            return
        
        # Create analysis window
        analysis_window = ctk.CTkToplevel(self.window)
        analysis_window.title("📋 OCR & NER Analysis Details")
        analysis_window.geometry("1200x800")
        
        # Center the window
        analysis_window.update_idletasks()
        x = (analysis_window.winfo_screenwidth() - 1200) // 2
        y = (analysis_window.winfo_screenheight() - 800) // 2
        analysis_window.geometry(f"1200x800+{x}+{y}")
        
        # Analysis content
        analysis_textbox = ctk.CTkTextbox(analysis_window, font=self.normal_font)
        analysis_textbox.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Build analysis content
        content = self.build_analysis_content()
        analysis_textbox.insert("1.0", content)
    
    def build_analysis_content(self):
        """Build detailed analysis content"""
        content = "🔍 OCR & NER ANALYSIS RESULTS\n"
        content += "=" * 80 + "\n\n"
        
        # OCR Results
        content += "📊 OCR RESULTS:\n"
        content += "-" * 40 + "\n"
        
        if self.ocr_results:
            content += f"Raw OCR Output (Combined): {self.ocr_text}\n\n"
            content += "Detailed OCR Results:\n"
            for i, result_group in enumerate(self.ocr_results):
                if result_group:
                    content += f"\nText Block {i+1}:\n"
                    for j, detection in enumerate(result_group):
                        bbox = detection[0]
                        text = detection[1][0]
                        confidence = detection[1][1]
                        content += f"  • Text: '{text}' (Confidence: {confidence:.2f})\n"
                        content += f"    Coordinates: {bbox}\n"
        else:
            content += "No text detected.\n"
        
        content += "\n" + "=" * 80 + "\n\n"
        
        # NER Results
        content += "🧠 NER RESULTS:\n"
        content += "-" * 40 + "\n"
        content += f"Input Text for NER: '{self.ocr_text}'\n\n"
        content += "Detected Entities:\n"
        
        if self.ner_results:
            for text, label in self.ner_results:
                content += f"  • '{text}' → {label}\n"
        else:
            content += "  No named entities detected.\n"
        
        content += f"\nFiltered Entities (COMPOSITION/COMMON NAME): {[text for text, label in self.ner_results if label in ['COMPOSITION NAME', 'COMMON NAME']]}\n"
        
        content += "\n" + "=" * 80 + "\n\n"
        
        # Custom Algorithm Information
        content += "🤖 CUSTOM ALGORITHM DETAILS:\n"
        content += "-" * 40 + "\n"
        content += "✨ TF-IDF & Cosine Similarity (Implemented from Scratch):\n"
        content += "   • Character-level n-grams (2-4 chars)\n"
        content += "   • Term Frequency: TF = (term_count / total_terms)\n"
        content += "   • Inverse Document Frequency: IDF = log(total_docs / docs_with_term)\n"
        content += "   • TF-IDF Vector: TF × IDF for each term\n"
        content += "   • Cosine Similarity: dot_product / (magnitude1 × magnitude2)\n"
        content += "   • No external libraries used (100% custom implementation)\n"
        
        return content
    
    def show_database_results(self):
        """Show database results in popup window"""
        if not self.database_results:
            messagebox.showinfo("Database", "No database results available.")
            return
        
        # Create database window
        db_window = ctk.CTkToplevel(self.window)
        db_window.title("💊 Database Search Results")
        db_window.geometry("1400x900")
        
        # Center the window
        db_window.update_idletasks()
        x = (db_window.winfo_screenwidth() - 1400) // 2
        y = (db_window.winfo_screenheight() - 900) // 2
        db_window.geometry(f"1400x900+{x}+{y}")
        
        # Header
        header_frame = ctk.CTkFrame(db_window, height=80, corner_radius=0)
        header_frame.pack(fill="x", padx=0, pady=0)
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="💊 Medicine Database Results",
            font=self.title_font,
            text_color="#059669"
        )
        title_label.pack(pady=25)
        
        # Results content
        results_textbox = ctk.CTkTextbox(db_window, font=self.normal_font)
        results_textbox.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        results_textbox.insert("1.0", self.database_results)
    
    def show_live_feed(self):
        """Show live camera feed with capture button inside"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera.")
            return
        
        # Create live feed window
        feed_window = ctk.CTkToplevel(self.window)
        feed_window.title("📹 Live Camera Feed")
        feed_window.geometry("720x650")
        
        # Center the window
        feed_window.update_idletasks()
        x = (feed_window.winfo_screenwidth() - 720) // 2
        y = (feed_window.winfo_screenheight() - 650) // 2
        feed_window.geometry(f"720x650+{x}+{y}")
        
        # Camera display
        camera_label = ctk.CTkLabel(feed_window, text="")
        camera_label.pack(expand=True, padx=20, pady=(20, 10))
        
        # Capture button inside live feed window
        capture_btn = ctk.CTkButton(
            feed_window,
            text="📸 Capture Image",
            font=self.normal_font,
            height=50,
            width=200,
            fg_color=self.button_color,
            hover_color=self.button_hover,
            command=lambda: self.capture_from_feed(feed_window)
        )
        capture_btn.pack(pady=(10, 20))
        
        def update_feed():
            ret, frame = self.cap.read()
            if ret:
                # Resize frame
                frame_resized = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Convert to tkinter format
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                camera_label.configure(image=photo)
                camera_label.image = photo
                
                # Schedule next update
                feed_window.after(10, update_feed)
            else:
                if self.cap:
                    self.cap.release()
                feed_window.destroy()
        
        def on_close():
            if self.cap:
                self.cap.release()
            feed_window.destroy()
        
        feed_window.protocol("WM_DELETE_WINDOW", on_close)
        update_feed()
    
    def capture_from_feed(self, feed_window):
        """Capture image from live feed"""
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Convert OpenCV image to PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Close feed window
                self.cap.release()
                feed_window.destroy()
                
                # Analyze the captured image
                self.analyze_image(pil_image)
    
    def show_info(self):
        """Show info dialog"""
        messagebox.showinfo(
            "About", 
            "Medical Info Provider v2.0\n\n"
            "AI-Powered Medicine Analysis System\n"
            "Built with CustomTkinter\n\n"
            "Features:\n"
            "• Advanced OCR with PaddleOCR\n"
            "• Named Entity Recognition\n"
            "• Custom Image Preprocessing (from scratch)\n"
            "• Custom TF-IDF & Cosine Similarity (from scratch)\n"
            "• Database Search with AI Matching\n"
            "• Modern UI with CustomTkinter\n\n"
            "🏆 Custom Algorithms Implemented:\n"
            "   ✓ Image Preprocessing Pipeline\n"
            "   ✓ TF-IDF Vectorization\n"
            "   ✓ Cosine Similarity Calculation\n\n"
            "© 2025 Medical Analysis Lab"
        )
    
    def run(self):
        """Start the application"""
        self.window.mainloop()

if __name__ == "__main__":
    app = MedicalInfoApp()
    app.run()
