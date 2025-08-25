# OCR

- PaddleOCR by PaddlePaddle is integrated to extract the text from images of medicine blister packet.

- Texts from all the images of dataset is extracted in a json file.

- PaddleOCR performs all the preprocessing necessary for text extraction.

- PaddleOCR uses detection algorithm DB and recognition algorithm SVTR_LCNet.

- OCR provides result of each image in a list with text, box coordinates and accuracy of the text prediction.