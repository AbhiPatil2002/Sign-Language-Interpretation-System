from string import ascii_uppercase
import numpy as np
import logging as lg
import cv2
import os
import operator
import tkinter as tk
from PIL import Image, ImageTk
import enchant
from keras.models import model_from_json
import time

class Application:
    def __init__(self):
        self.hs = enchant.Dict("en_US")  # Initialize enchant dictionary for English (US)
        self.vs = cv2.VideoCapture(0)  # Initialize video capture object (webcam)
        self.current_image = None
        self.current_image2 = None

        # Load the pre-trained model for general predictions
        self.json_file = open(r"C:/Users/ABHISHEK PATIL/Mega Project/Sign-Language-To-Text-Conversion-main/Models/model_new2.json")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights("C:/Users/ABHISHEK PATIL/Mega Project/Sign-Language-To-Text-Conversion-main/Models/model_new.h5")

        # Load the pre-trained model for D, R, U predictions
        self.json_file_dru = open(r"C:/Users/ABHISHEK PATIL/Mega Project/Sign-Language-To-Text-Conversion-main/Models/model-bw_dru.json")
        self.model_json_dru = self.json_file_dru.read()
        self.json_file_dru.close()
        self.loaded_model_dru = model_from_json(self.model_json_dru)
        self.loaded_model_dru.load_weights("C:/Users/ABHISHEK PATIL/Mega Project/Sign-Language-To-Text-Conversion-main/Models/model-bw_dru.h5")

        # Load the pre-trained model for D, I, K, T predictions
        self.json_file_tkdi = open(r"C:/Users/ABHISHEK PATIL/Mega Project/Sign-Language-To-Text-Conversion-main/Models/model-bw_tkdi.json")
        self.model_json_tkdi = self.json_file_tkdi.read()
        self.json_file_tkdi.close()
        self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        self.loaded_model_tkdi.load_weights("C:/Users/ABHISHEK PATIL/Mega Project/Sign-Language-To-Text-Conversion-main/Models/model-bw_tkdi.h5")

        # Load the pre-trained model for M, N, S predictions
        self.json_file_smn = open(r"C:/Users/ABHISHEK PATIL/Mega Project/Sign-Language-To-Text-Conversion-main/Models/model-bw_smn.json")
        self.model_json_smn = self.json_file_smn.read()
        self.json_file_smn.close()
        self.loaded_model_smn = model_from_json(self.model_json_smn)
        self.loaded_model_smn.load_weights("C:/Users/ABHISHEK PATIL/Mega Project/Sign-Language-To-Text-Conversion-main/Models/model-bw_smn.h5")

        self.ct = {'blank': 0}  # Initialize a dictionary to store character counts
        for i in ascii_uppercase:
            self.ct[i] = 0  # Set initial count for each character to 0

         

        lg.basicConfig(level=lg.DEBUG)

        print("Loaded model from disk")

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1760x990")

        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)

        self.panel2 = tk.Label(self.root)  # Initialize image panel
        self.panel2.place(x=400, y=65, width=275, height=275)

        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="Sign Language To Text Conversion", font=("Courier", 30, "bold"))

        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=400, y=540)

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=540)
        self.T1.config(text="Character :", font=("Courier", 30, "bold"))

        self.panel4 = tk.Label(self.root)  # Word
        self.panel4.place(x=220, y=595)

        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=595)
        self.T2.config(text="Word :", font=("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root)  # Sentence
        self.panel5.place(x=350, y=645)

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=645)
        self.T3.config(text="Sentence :", font=("Courier", 30, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x=250, y=690)
        self.T4.config(text="Suggestions :", fg="red", font=("Courier", 30, "bold"))

        self.bt1 = tk.Button(self.root, command=self.action1, height=0, width=0)
        self.bt1.place(x=26, y=745)

        self.bt2 = tk.Button(self.root, command=self.action2, height=0, width=0)
        self.bt2.place(x=325, y=745)

        self.bt3 = tk.Button(self.root, command=self.action3, height=0, width=0)
        self.bt3.place(x=625, y=745)

           # Add a new button for detecting the current word
        self.detect_word_button = tk.Button(self.root, text="Detect", command=self.detect_current_word, font=("Courier", 20))
        self.detect_word_button.place(x=650, y=550)
         # Link the new function to the button
        self.detect_word_button.config(command=self.detect_current_word)

        self.bt_clear = tk.Button(self.root, text="Clear", command=self.clear_word, font=("Courier", 20))
        self.bt_clear.place(x=800, y=550)

        

        self.str = ""  # Initialize an empty string for the sentence
        self.word = ""  # Initialize an empty string for the current word
        self.current_symbol = "Empty"  # Initialize the current symbol to "Empty"
        self.photo = "Empty"
        self.blank_flag = 0  # Initialize the blank flag to 0
        self.video_loop()  # Start the video loop

    def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            cv2image = cv2.flip(frame, 1)

            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            cv2image = cv2image[y1: y2, x1: x2]

            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            self.predict(res)

            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)

            self.panel3.config(text=self.current_symbol, font=("Courier", 30))
            self.panel4.config(text=self.word, font=("Courier", 30))
            self.panel5.config(text=self.str, font=("Courier", 30))

            if self.word:  # Check if self.word is not an empty string
                predicts = list(self.hs.suggest(self.word))
            else:
                predicts = []  # Set predicts to an empty list

            if len(predicts) > 0:
                self.bt1.config(text=predicts[0], font=("Courier", 20))
            else:
                self.bt1.config(text="")

            if len(predicts) > 1:
                self.bt2.config(text=predicts[1], font=("Courier", 20))
            else:
                self.bt2.config(text="")

            if len(predicts) > 2:
                self.bt3.config(text=predicts[2], font=("Courier", 20))
            else:
                self.bt3.config(text="")

        self.root.after(5, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))  # Resize the test image to 128x128 pixels

        # Make predictions using the general model
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        prediction = {'blank': result[0][0]}
        for i in range(min(len(result[0]), len(ascii_uppercase))):
            prediction[ascii_uppercase[i]] = result[0][i + 1]

        # Determine the current symbol based on the highest prediction probability
        max_prob_index = np.argmax(result[0])

        if max_prob_index < len(ascii_uppercase):
            self.current_symbol = ascii_uppercase[max_prob_index]
        else:
            self.current_symbol = 'blank'

        lg.debug(f"General model prediction probabilities: {prediction}")
        lg.debug(f"Selected current symbol: {self.current_symbol}")

        # Sort predictions and update current_symbol
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        # Make predictions using the D, R, U model
        if self.current_symbol in ['D', 'R', 'U']:
            result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1))
            prediction_dru = {'D': result_dru[0][0], 'R': result_dru[0][1], 'U': result_dru[0][2]}
            prediction_dru = sorted(prediction_dru.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction_dru[0][0]

            lg.debug(f"DRU model prediction probabilities: {prediction_dru}")

        # Make predictions using the D, I, K, T model
        if self.current_symbol in ['D', 'I', 'K', 'T']:
            result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1))
            prediction_tkdi = {'D': result_tkdi[0][0], 'I': result_tkdi[0][1], 'K': result_tkdi[0][2], 'T': result_tkdi[0][3]}
            prediction_tkdi = sorted(prediction_tkdi.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction_tkdi[0][0]

            lg.debug(f"DIKT model prediction probabilities: {prediction_tkdi}")

        # Make predictions using the M, N, S model
        if self.current_symbol in ['M', 'N', 'S']:
            result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1))
            prediction_smn = {'M': result_smn[0][0], 'N': result_smn[0][1], 'S': result_smn[0][2]}
            prediction_smn = sorted(prediction_smn.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction_smn[0][0]

            lg.debug(f"MNS model prediction probabilities: {prediction_smn}")

        lg.debug(f"Final current symbol after all models: {self.current_symbol}")

        # Reset character counts and handle blank flag and other logic as needed
        if self.current_symbol == 'blank':
            self.reset_character_counts()
        else:
            self.ct[self.current_symbol] += 1  # Increment the count for the current symbol

        if self.ct[self.current_symbol] > 60:  # If the count for the current symbol exceeds 60
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                difference = abs(self.ct[self.current_symbol] - self.ct[i])
                if difference <= 20:  # If the difference between the counts is less than or equal to 20
                    self.reset_character_counts()
                    return

            self.reset_character_counts()

    def reset_character_counts(self):
        # Function to reset character counts and blank flag
        self.ct = {char: 0 for char in ascii_uppercase}
        self.ct['blank'] = 0
        self.blank_flag = 0

    # Define the function that will be called when the button is clicked
    def detect_current_word(self):
        # Append the current symbol (character) to the current word only if it's not "Empty"
        if self.current_symbol != "Empty" and self.current_symbol != "blank":
            self.word += self.current_symbol  # Add the current symbol to the word
            
            # Reset the current symbol to "Empty" after adding it to the word
            self.current_symbol = "Empty"
        
        # Update the word label to reflect the updated word
        self.panel4.config(text=self.word, font=("Courier", 30))

        time.sleep(0.1)

    def clear_word(self):
        self.word = ""
        self.panel4.config(text=self.word, font=("Courier", 30))



       

    def action1(self):
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 0:
            self.word = ""
            self.str += " "
            self.str += predicts[0]  # Add the first suggestion to the sentence

    def action2(self):
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 1:
            self.word = ""
            self.str += " "
            self.str += predicts[1]  # Add the second suggestion to the sentence

    def action3(self):
        predicts = self.hs.suggest(self.word)
        if len(predicts) > 2:
            self.word = ""
            self.str += " "
            self.str += predicts[2]  # Add the third suggestion to the sentence

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
        

print("Starting Application...")
app = Application()
(app).root.mainloop()