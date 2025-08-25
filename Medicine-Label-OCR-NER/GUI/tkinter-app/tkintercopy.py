# !/usr/bin/python3

import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt
import PIL 
import os
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import mysql.connector
from tkinter import ttk
from tkinter import scrolledtext
import cv2
from tkinter import messagebox


cap =   None
#model import
ocr_model = PaddleOCR(lang='en', use_gpu=False)

#main window definition
my_w = tk.Tk()
my_w.geometry("1000x800")  # Size of the window 
my_w.title('MEDICAL INFO PROVIDER')
Title_font=('Ubuntu', 20, 'bold')
Normal_font=('Ubuntu' , 14 )
scroll_font= ('Ubuntu' , 10)

#background image 
bg = ImageTk.PhotoImage(file = "../pharms.jpg") 
background_label = tk.Label(my_w, image=bg)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

#design & button
l1 = tk.Label(my_w,text='Upload the medicine image',font=Normal_font)  
l1.grid(row=3,column =1 )
l2 = tk.Label(my_w , text = "Medicine Label Extraction System" , font=  Title_font)
l2.grid(row=2, column =1 , pady=100 , padx = 290 )
b1 = tk.Button(my_w, text='Upload File', 
   width=20,command = lambda:app("upload"))
b1.grid(row=4,column=1 , pady= 30) 
b5 = tk.Button(my_w, text='Capture Image', 
   width=20,command = lambda:app("capture"))
b5.grid(row=6,column=1,pady=30)
b6 = tk.Button(my_w , text = 'Open Livefeed' , 
               width=20 , command = lambda:show())
b6.grid(row = 5 ,column=1,pady=5) 

def show():
    global cap
    if cap is not None:
        cap.release()

    # Create a new camera object
    cap = cv2.VideoCapture(0)
  
    live_window = tk.Toplevel(my_w)
    live_window.title('Live Feed')
    live_window.geometry("640x480")
    
    canvas = tk.Canvas(live_window, width=640, height=480)
    canvas.pack()

    def update():
        ret, frame = cap.read()
        if ret:
            # Convert the OpenCV BGR image to RGB format for Tkinter
            
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the canvas with the new image
            canvas.imgtk = imgtk
            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

            # Schedule the next update after 10 milliseconds
            live_window.after(10, update)
        else:
            # Handle the case when the video capture fails
            cap.release()
            live_window.destroy()

    # to update function initially
    update()


def app(click):
    global cap
    #to capture image
    if click == "capture": 
      ret, frame = cap.read()
      if not ret or frame is None:
           messagebox.showerror("Error", "Camera could not be found")
           return
      cap.release()
      frame = cv2.resize(frame, (400, 400))
      img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      img = Image.fromarray(img)
      ocr_img = img 
      img = ImageTk.PhotoImage(img)

    #to upload image 
    if click == "upload":
      f_types=[('all files', '.*'),
               ('text files', '.txt'),
               ('image files', '.png'),
               ('image files', '.jpg'),
               ('image files', '.jpeg'),
               ]
      filename = filedialog.askopenfilename(filetypes=f_types)
      img=Image.open(filename)
      ocr_img = Image.open(filename)

    # new width & height
      img_resized=img.resize((400,400)) 
      img=ImageTk.PhotoImage(img_resized)

    #save the image in a folder called 'upload images'
    save_folder = '../uploaded_images/'
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, 'med.jpg')
    ocr_img.save(save_path)

    #create a new window 
    new_window = tk.Toplevel(my_w)
    new_window.title('Uploaded Image')
    new_window.geometry("1000x1000")
    background_label = tk.Label(new_window, image=bg)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    image_label = tk.Label(new_window, image=img)
    image_label.image = img
    image_label.grid( row = 1, column=1 , pady=50 , padx = 250 )
    
    #apply ocr 
    result = ocr_model.ocr(os.path.join('../uploaded_images/med.jpg'))
    if result:  
      output=" "
      for res in result:
        if res:
          for re in res:
            output += re[1][0] + " "
    #l4 = tk.Label(new_window,text= output,font=Normal_font)  
    #l4.grid(row=3,column =1 )
    #print(output)
    else:
       result = "null"
    #import model
    ent_list=[]
    nlp_ner = spacy.load("../model-best")
    doc = nlp_ner(output)
    for ent in doc.ents:
      if ent.label_ == 'COMPOSITION NAME' or 'COMMON NAME' :
         ent_list.append(ent.text)
    if ent_list: 
       
   #store name 
      ores= f"Medicine name: {ent_list[0]} "

      med_name = ent_list[0] 
      print(med_name)

      fsecword = ""
      if len(ent_list) > 1:
        med_sec_name = ent_list[1]
        fsecword = med_sec_name.split()[0]
        print(fsecword)

      fword = med_name.split()[0]
      print(fword)
      
      if ores: 
        #l3 = tk.Label(new_window,text= ores, font=Normal_font)  
        #l3.grid(row=3,column =1 )

        #database connection
        conn=mysql.connector.connect(host="localhost",username="root",password="dbmsxaxm**0",database="medicine")
        my_cursor=conn.cursor()

        #data query
        query = "SELECT * FROM med_info WHERE LOWER(DRUG_NAME) LIKE CONCAT('%', LOWER(%s), '%')"
        my_cursor.execute(query, (fword.lower(),)) 
        data = my_cursor.fetchall()

        if not data and fsecword:
          query = "SELECT * FROM med_info WHERE LOWER(DRUG_NAME) LIKE CONCAT('%', LOWER(%s), '%')"
          my_cursor.execute(query, (fsecword.lower(),))
          data = my_cursor.fetchall()
            
        column_labels = [ "DRUG NAME", "DOSAGE FORM AND STRENGTH", "INDICATIONS","CONTRAINDICATIONS OR PRECAUTIONS", "DOSAGE SCHEDULE", "ADVERSE EFFECTS", "DRUG AND FOOD INTERACTIONS"]

        # Display the data in a scrolledtext widget
        wrap_length = 200  
        text_widget = scrolledtext.ScrolledText(new_window, wrap=tk.WORD, width=120, height=27 , bg="white", fg="black", font = scroll_font)
        text_widget.grid(row=2, column=1, columnspan=len(column_labels) + 1, pady=10, padx=30)

        for row in data:
              for j, value in enumerate(row):
                if value: 
                  label_text = f"{column_labels[j]}: {value}\n\n"
                  text_widget.insert(tk.END, label_text )

        text_widget.config(state=tk.DISABLED)
        
        my_cursor.close()
        conn.close()

    else: 
      l3 = tk.Label(new_window,text= "Sorry couldn't provide the result, check the following:", font=Normal_font)  
      l3.grid(row=3,column =1 )
      rule_list = ["1. Image is clear.", "2. Image is of blister pack and not other medicine packaging."]
      wrap_length = 200  
      text_output = scrolledtext.ScrolledText(new_window, wrap=tk.WORD, width=120, height=3, bg="black", fg="red", font = scroll_font)
      text_output.grid(row=4, column=1, columnspan=len(rule_list) + 1, pady=10, padx=30)
      for item in rule_list:
    	  text_output.insert(tk.END, item + "\n") 
      
      text_output.config(state=tk.DISABLED)



   
    new_window.resizable(0 , 0)

my_w.resizable(0 , 0)

my_w.mainloop()  # Keep the window open   

