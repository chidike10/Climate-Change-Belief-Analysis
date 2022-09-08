"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
import numpy as np                     
import pandas as pd

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("michael_vector.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("train.csv")

def app_output (pn):   
	if (pn == -1):
		pr=("Pro: the tweet supports the belief of man-made climate change") 
	elif (pn == 0):
		pr=("News: the tweet links to factual news about climate change")                  
	elif (pn == 1):
		pr=("Neutral: the tweet neither supports nor refutes the belief of man-made climate change")
	else:
		pr=("Anti: the tweet does not believe in man-made climate change") 
	return st.text(pr) 

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Tweet Classifer")
	#st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	st.sidebar.image('logt.jpeg', width =150)
	st.sidebar.title("ZF3 DATA BOT®")
#	st.sidebar.subheader("Defining growth through data")
	st.sidebar.title("Menu")
    
    
       
#	st.sidebar.subheader("Defining growth through data")
#	st.sidebar.title("Menu")
#	st.image('bar_sentiments.jpg')     
	options = ["Prediction", "EDA", "Information", "About the company"]
	selection = st.sidebar.selectbox("Choose Option", options)
    

    
	if selection == "EDA":
		st.title("Exploratory Data Analysis")
#	with st.container():
#	st.write("model performance")

        # You can call any Streamlit command, including custom components:
#			st.bar_chart(np.random.randn(50, 3))

#		st.write("model not to scale")
		st.subheader("Climate Wordcloud")  
		st.image('climate_wordcloud.jpg')

		st.subheader("Bar Graph representation of the models")        
		st.image('der.jpg')
        
		st.subheader("Twitter sentiments")
		st.image('twittersentiment.jpg')        
#		container = st.container()
#		container.write("This is inside the container")
#		st.write("This is outside the container")

		# Now insert some more in the container
#		container.write("This is inside too")
    
    
	# Building out the "Information" page
	if selection == "Information":
		st.title("Information Page")
		st.subheader("Mission Statement")
		st.info(" To Provide an accurate and robust solution to companies to access a broad base of consumer sentiment, spanning multiple demographic and geographic categories hence increasing their insights and informing future marketing strategies.")
		st.subheader("Vision Statement")
		st.info("A model that is able to classify whether or not a person believes in climate change, based on their novel tweet data")
#	st.markdown("A model that is able to classify whether or not a person believes in climate change, based on their novel tweet data")        
		st.subheader("Confusion Matrix of the Logistric Regression Model")
		st.image('cf.jpg')
		st.subheader("Bar Graph showing the size of the individual sentiments")
		st.image("bars.jpg")
		# You can read a markdown file from supporting resources folder

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
            
            
            
            
            

	# Building out the predication page
	if selection == "Prediction":
		st.title("ABC Plastic")     
		st.subheader("Climate change tweet classifier")
#		st.info("Please Enter your text below and click  'classify'  to output results")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter your Text below to classify","Type Here")        
        
        # you can create multiple pages this way
		options = ["Logistic Regression Model", "Naive bayes model", "Passive model" ,"SVC model"]
		selection = st.sidebar.selectbox("Select Model", options)

		if st.button("Click to Classify"):
		# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
		# Load your .pkl file with the model of your choice + make predictions
		# Try loading in multiple models to give the user a choice

    
			if selection == "Logistic Regression Model":  
				predictor = joblib.load(open(os.path.join("michael_log_reg_model.pkl"),"rb"))
				a = predictor.predict(vect_text)           
				app_output(a)                       
                                        
			if selection == "Naive bayes model":
				predictor = joblib.load(open(os.path.join("michael_naive_bayes_model.pkl"),"rb"))
				b = predictor.predict(vect_text)
				app_output(b) 
               
			if selection == "Passive model":  
				predictor = joblib.load(open(os.path.join("michael_passive_model.pkl"),"rb"))
				c = predictor.predict(vect_text)
				app_output(c) 
                                         
			if selection == "SVC model":  
				predictor = joblib.load(open(os.path.join("michael_SVC_model.pkl"),"rb"))
				d = predictor.predict(vect_text)                   
				app_output(d) 
               
	# Building our company's profile page
	if selection == "About the company":
		st.title("Company's profile") 
#		st.subheader("Welcome to ZF3 company")
		st.info("Welcome to ZF3 company. The company was founded in June 2022 by the following pioneers")
		# You can read a markdown file from supporting resources folder

        
#Display Images side by side        
		from PIL import Image        
		col1, col2 = st.columns(2)

		with col1:
			st.header("Harmony Odumuko")
			st.subheader("President")             
			st.image('Harmony Odumuko.jpg', width =243)

		with col2:
			st.header("Michael Mamah")
			st.subheader("Vice-President")             
			st.image('Michael Mamah.jpg')
                                   
		col3, col4 = st.columns(2)
        
		with col3:
			st.header("Raheemat Adetunji")
			st.subheader("Cloud expert")            
			st.image('Raheemat Adetunji.jpg', width=190)
            
		with col4:
			st.header("Abigael Kinini")
			st.subheader("Director Strategies ")             
			st.image('Abigael Kinini.jpg', width =212)               
        
		col5, col6 = st.columns(2)     
        
		with col5:
			st.header("Victor Meleka")
			st.subheader("Technical Operations")             
			st.image('Victor Meleka.jpg', width =265)   
                      
		with col6:           
			st.header("Francis Ikegwu")
			st.subheader("Business Intelligence")             
			st.image('Francis Ikegwue.jpg', width=250)        
        


        #image doc code format

#		from PIL import Image
#		image1 = Image.open('abcf.jpg')        
#		victor_image = Image.open('victor.jpg')
#		Francis_image = Image.open('abcf.jpg')        
#		Abigael_image = Image.open('abcf.jpg')
#		Michael_image = Image.open('abcf.jpg')
#		Raheeemat_image = Image.open('abcf.jpg')
#		Harmony_image = Image.open('abcf.jpg') 
        
#		st.image(image1, caption='Sunrise by the mountains', width=350)
#		st.image(victor_image, caption='Model Expert', width=350) 
        
        #documentation for images
        #st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

      
		st.subheader("More information")
		if st.checkbox('Show contact information'): # data is hidden if box is unchecked
			st.info("francisikegwu@yahoo.com, kininiabigael@gmail.com, mamahchidike@gmail.com,icontola@gmail.com, vicmeleka@gmail.com, nibotics@gmail.com")
            # will write the df to the page  


#		st.markdown("Francis Ikegwu,Abigael Kinini, Michael Mamah, Raheeemat Adetunji,Victor Meleka, Harmony Odumuko") 

#		st.subheader("More information")
#		if st.checkbox('Show contact information'): # data is hidden if box is unchecked
#			st.info("francisikegwu@yahoo.com, kininiabigael@gmail.com,mamahchidike@gmail.com, icontola@gmail.com, vicmeleka@gmail.com, nibotics@gmail.com") # will write the df to the page  
        # video format /doc
        
#st.bar_chart({"data": [1, 5, 2, 6, 2, 1]})

		with st.expander("Expand to see Company's video profile"):
#		st.write("""
#         The chart above shows some numbers I picked for you.
#         I rolled actual dice for these, so they're *guaranteed* to
#         be random.
#     """)
#		st.image("https://static.streamlit.io/examples/dice.jpg")        
			video_file = open('Kinini.mp4', 'rb')
			video_bytes = video_file.read()
			st.video(video_bytes)            
	st.sidebar.subheader("Defining growth through data")            

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			#st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
