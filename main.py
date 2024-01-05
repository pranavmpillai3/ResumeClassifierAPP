import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QFileDialog, QPushButton, QTextEdit, QComboBox
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pytesseract
from PIL import Image


class ResumeClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.MAX_FEATURES = 200000
        self.vectorizer = TextVectorization(max_tokens=self.MAX_FEATURES,
                                            output_sequence_length=1800,
                                            output_mode='int')
        self.df = pd.read_csv('UpdatedResumeDataSet.csv', index_col=None)
        self.labelencoder = LabelEncoder()
        self.df['Category'] = self.labelencoder.fit_transform(self.df['Category'])
        self.label_mapping = dict(zip(self.labelencoder.classes_, self.labelencoder.transform(self.labelencoder.classes_)))
        self.X = self.df['Resume']
        self.y = self.df['Category']
        self.y = np.array(self.y)
        self.y_encoded = to_categorical(self.y, num_classes=len(self.df['Category'].value_counts()))
        self.vectorizer.adapt(self.X.values)
        self.vectorized_text = self.vectorizer(self.X.values)

    def init_ui(self):
        self.setWindowTitle('Resume Classifier')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        self.resume_input = QTextEdit(self)
        self.resume_input.setPlaceholderText('Enter Skills Description..')
        layout.addWidget(self.resume_input)

        image_button = QPushButton('Select Resume', self)
        image_button.clicked.connect(self.select_image)
        layout.addWidget(image_button)

        check_button = QPushButton('Check', self)
        check_button.clicked.connect(self.check_description)
        layout.addWidget(check_button)

        reset_button = QPushButton('Reset', self)
        reset_button.clicked.connect(self.reset_description)
        layout.addWidget(reset_button)

        exit_button = QPushButton('Exit', self)
        exit_button.clicked.connect(self.exit_program)
        layout.addWidget(exit_button)

        self.result_label = QLabel('', self)
        layout.addWidget(self.result_label)
        self.setLayout(layout)


    # def scrape_from_google(self):
    #     selected_category = self.skill_dropdown.currentText()
    #     if selected_category == 'Select skill to get from google':
    #         return
    #
    #     search_query = f'{selected_category}+skills'
    #     url = f'https://www.google.com/search?q={search_query}'
    #
    #     try:
    #         response = requests.get(url)
    #         print(response.content)
    #         soup = BeautifulSoup(response.text, 'html.parser')
    #         search_results = soup.find_all('div', class_='RqBzHd')
    #         print(search_results)
    #         scraped_text = '\n'.join([result.get_text() for result in search_results])
    #         print(scraped_text)
    #         self.resume_input.setPlainText(scraped_text)
    #     except requests.exceptions.RequestException as e:
    #         print('Error')

    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.bmp *.gif *.jpeg)", options=options)
        if image_path:
            self.extract_text_from_image(image_path)


    def extract_text_from_image(self, image_path):

        try:
            image = Image.open(image_path)
            extracted_text = pytesseract.image_to_string(image)
            self.resume_input.setPlainText(extracted_text)
        except Exception as e:
            print(f'error extracting text from image : {e}')



    def check_description(self):
        model = tf.keras.models.load_model('resume_classification.h5')
        resume_description = self.resume_input.toPlainText().lower()
        print(resume_description)
        vectorized_description = self.vectorizer([resume_description])
        result = model.predict(vectorized_description)
        print(result)
        print(np.argmax(result[0]))
        print(result[0][np.argmax(result[0])])
        for category, idx in self.label_mapping.items():
            if idx == np.argmax(result[0]):
                print(category)
                self.result_label.setText(category)

    def reset_description(self):
        self.resume_input.clear()
        self.result_label.clear()

    def exit_program(self):
        sys.exit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ResumeClassifier()
    window.show()
    sys.exit(app.exec_())












