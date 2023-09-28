from distutils.command.clean import clean
import pickle
import re
import nltk
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')

# loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

def textcleaning(text):
    import string
    from textblob import TextBlob
    exclude = string.punctuation
    
    cleantext = text.lower()     # lower the string
    cleantext = re.sub("<.*?>", " ", cleantext)   # Removing the HTML tags
    cleantext = re.sub("https?://\S+|www\.\S+"," ",cleantext)   # Removing the URLS's
    cleantext = cleantext.translate(str.maketrans('','',exclude)) # Removing the Punctuations
    cleantext = re.sub("\s+"," ",cleantext) # Removing the extra characters (\r\n)
    #cleantext = TextBlob(cleantext)
    #cleantext = cleantext.correct().string  # Removing the Spelling mistake
    return cleantext

# web app
def model():
    st.title("Resume Screening App")
    upload_file = st.file_uploader("Upload Resume",type=['txt','pdf'])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')


        clean_resume = textcleaning(resume_text)
        clean_resume = tfidf.transform([clean_resume])
        prediction_id = clf.predict(clean_resume)[0]
        #st.write(prediction_id)
        
        category_mapping = {
            15: "Java Developer",
            23: "Manual Testing Engineer",
            8: "Devops Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop Engineer",
            3: "Blockchain Engineer",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Scientist",
            22: "Sales Manager",
            16: "Mechanical Engineer",
            1: "Artist",
            7: "Database Administrator",
            11: "Electrical Engineer",
            14: "Health and Fitness Expert",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing Engineer",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate"
            }

        category_name = category_mapping.get(prediction_id,"Unknown")

        st.write("Predicted Category:", category_name)


# python main
if __name__ == "__main__":
    model()