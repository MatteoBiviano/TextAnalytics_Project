import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from collections import Counter

# Set background
def set_background(is_set, text = None):
	if is_set:
		if text:
			header_html = f"""
				<div style="background-color:#145796;padding:10px;border-radius:10px">
				<h1 style="color:white;text-align:center;">{text}</h1>
				</div>
			"""
			components.html(header_html)

		image = Image.open('images/background.png')
		st.image(image)

# Load Dataframe
@st.cache
def load_data(path, encoding = None):
	return pd.read_csv(path, encoding = encoding)


# Show random tweets
def show_r_tweet(data, st):
	st.subheader("Which sentiment do you want to select?")
	sentiment = st.selectbox(
		'', 
		["", "Positive", "Neutral", "Negative"])

	if not sentiment == "":

		st.subheader(f'Random tweet for "{sentiment}" sentiment')

		text = data.query("airline_sentiment == @sentiment.lower()")[["text"]].sample(n=1).iat[0, 0]


		html_temp = f"""
			<div style="background-color:#145796;padding:30px;border-radius:30px">
				<h1 style="color:white;text-align:center;">{text}</h1>
			</div>
			"""
		components.html(html_temp, height = 300)
	else:
		set_background(True, text = None)

# Number of tweets by sentiment
def show_bysentiment(data, st):
	st.subheader("Number of Tweets by Sentiment")
	chart = st.selectbox(
		'How do you want to visualize the sentiment?',
		["Bar Plot", "Pie Chart"])

	sentiment_count = data["airline_sentiment"].value_counts()
	sentiment_count = pd.DataFrame({"Sentiment":sentiment_count.index, "Number of tweets":sentiment_count.values})
	
	if(chart == "Bar Plot"):
		fig = px.bar(sentiment_count, x="Sentiment", y="Number of tweets", color_continuous_scale="bluyl", color="Number of tweets")
		st.plotly_chart(fig)
	else:
		fig = px.pie(sentiment_count, values="Number of tweets", names="Sentiment")
		st.plotly_chart(fig)


# Number of tweets by Airline
def show_byairline(data, st):
	st.subheader("Number of Tweets by Airline")
	airline_chart = st.selectbox(
		'How do you want to visualize the airlines?',
		["Bar Plot", "Pie Chart"], index = 0)
	sentiment_count = data["airline"].value_counts()
	sentiment_count = pd.DataFrame({"Airline company":sentiment_count.index, "Number of tweets":sentiment_count.values})
	
	if(airline_chart == "Bar Plot"):
		fig = px.bar(sentiment_count, x="Airline company", y="Number of tweets", color_continuous_scale="bluyl",  color="Number of tweets")
		st.plotly_chart(fig)
	else:
		fig = px.pie(sentiment_count, values="Number of tweets", names="Airline company")
		st.plotly_chart(fig)


# Plot tweets' location map
def tweet_map(data, st):

	st.subheader("Tweets' location map by airline")
	multi_airline = st.multiselect(
		"Select the airlines for which to plot the tweets' location map",
		list(set(data["airline"])))
	if(len(multi_airline)>0):
		coords = data[data["airline"].isin(multi_airline)]["tweet_coord"]
	else:
		coords = data["tweet_coord"]
	not_na_index = list(coords.dropna().index)
	latitude = []
	longitude = []
	for i in not_na_index:
		latitude.append(float(coords[i].split(",")[0][1:]))
		longitude.append(float(coords[i].split(",")[1][1:-1]))
	dt = pd.DataFrame({"latitude":latitude, "longitude":longitude})
	st.map(dt)


# Number of negative_reason by Airline
def negatives_byairline(data, st):
	st.subheader("Number of negative reasons by Airline")

	neg_reason = list(set(data["negativereason"]))

	fig = go.Figure()
	for i in set(data["airline"]):
		y_val = []
		for reas in neg_reason:
			y_val.append(len(data[(data["airline"] == i) & (data["negativereason"] == reas)]))
		fig.add_trace(go.Bar(
		    x=neg_reason,
		    y=y_val,
		    name=i,
		))

	fig.update_layout(barmode='group', 
						xaxis_tickangle=-60, 
						autosize=False, width=900,
						height=500)
	st.plotly_chart(fig)


# Number of negative_reason by Airline
def airline_bysentiment(data, st):
	st.subheader("Number of sentiments by Airline")

	airlines = list(set(data["airline"]))

	fig = go.Figure()
	for i in set(data["airline_sentiment"]):
		y_val = []
		for airl in airlines:
			y_val.append(len(data[(data["airline_sentiment"] == i) & (data["airline"] == airl)]))
		fig.add_trace(go.Bar(
		    x=airlines,
		    y=y_val,
		    name=i,
		))

	fig.update_layout(barmode='group', 
						xaxis_tickangle=-60, 
						autosize=False, width=700,
						height=500)
	st.plotly_chart(fig)

# Show negative reason count by datetime and airline
def negative_bydate(data, st):
	st.subheader("Negative reasons by DateTime")
	date = data.copy().reset_index()
	date.tweet_created = pd.to_datetime(date.tweet_created)
	date.tweet_created = date.tweet_created.dt.date
	df = date
	day_df = df.groupby(['tweet_created','airline','airline_sentiment', "negativereason"]).size()
	day_df = day_df.loc(axis=0)[:,:,'negative']

	#groupby and plot data
	grouped = day_df.groupby(['tweet_created','airline']).sum().unstack()


	dates_col = []
	for date in grouped.index:
		dates_col.append(str(date))

	fig = go.Figure()
	for i in grouped.columns:
		y_val = list(grouped[i].values)
		fig.add_trace(go.Bar(
		    x=dates_col,
		    y=y_val,
		    name=i,
		))

	fig.update_layout(barmode='group', 
						xaxis_tickangle=-60, 
						autosize=False, width=800,
						height=500)
	
	st.plotly_chart(fig)

	st.subheader("Select date and airline to view the number of tweets for negative reason")

	date_time_chart = st.selectbox(
		'Which data do you want to view?',
		[""] + dates_col, index = 0)
	airline_chart = st.selectbox(
		'Which airline do you want to view?',
		[""] + list(set(data["airline"])), index = 0)

	if len(date_time_chart) > 0 and len(airline_chart) > 0:
		sentiment_count = df[(df["tweet_created"]== pd.to_datetime(date_time_chart)) & (df["airline"]==airline_chart)]["negativereason"].value_counts()
		sentiment_count = pd.DataFrame({"Negative reason":sentiment_count.index, "Number of tweets":sentiment_count.values})
		fig = px.bar(sentiment_count, x="Negative reason", y="Number of tweets", color_continuous_scale="bluyl", color="Number of tweets")
		fig.update_layout(xaxis_tickangle=-65)
		st.plotly_chart(fig)

# Select type of data analysis
def switch_data_analysis(analysis_code, data, st):
	if analysis_code =="1":
		show_r_tweet(data, st)
	elif analysis_code == "2":
		show_bysentiment(data, st)
	elif analysis_code == "3":
		show_byairline(data, st)
	elif analysis_code == "4":
		tweet_map(data, st)
	elif analysis_code == "5":
		negatives_byairline(data, st)
	elif analysis_code == "6":
		airline_bysentiment(data, st)
	elif analysis_code == "7":
		negative_bydate(data, st)


# Character frequency count by Sentiment
def character_freq_count(data, st):
	st.subheader("Character frequency count by Sentiment")

	neg = data[data['airline_sentiment']=='negative']['not_tag_text'].str.len()
	pos = data[data['airline_sentiment']=='positive']['not_tag_text'].str.len()
	neu = data[data['airline_sentiment']=='neutral']['not_tag_text'].str.len()

	fig = make_subplots(rows=1, cols=3)

	fig.add_trace(
		go.Histogram(x=list(neg), name='Negative Tweets'),
		row=1, 
		col=1
	)

	fig.add_trace(
		go.Histogram(x=list(pos), name='Positive Tweets'),
		row=1, 
		col=2,
	)

	fig.add_trace(
		go.Histogram(x=list(neu), name='Neutral Tweets'),
		row=1, 
		col=3,
	)

	fig.update_layout(height=400, width=800)

	st.plotly_chart(fig)

# Word Count Distribution
def word_count_distribution(data, st):
	st.subheader("Word Count Distribution")

	neg = data[data['airline_sentiment']=='negative']['not_tag_text'].str.split().map(lambda x: len(x))
	pos = data[data['airline_sentiment']=='positive']['not_tag_text'].str.split().map(lambda x: len(x))
	neu = data[data['airline_sentiment']=='neutral']['not_tag_text'].str.split().map(lambda x: len(x))

	fig = make_subplots(rows=1, cols=3)

	fig.add_trace(
		go.Histogram(x=list(neg), name='Negative Tweets'),
		row=1, 
		col=1
	)

	fig.add_trace(
		go.Histogram(x=list(pos), name='Positive Tweets'),
		row=1, 
		col=2,
	)

	fig.add_trace(
		go.Histogram(x=list(neu), name='Neutral Tweets'),
		row=1, 
		col=3,
	)

	fig.update_layout(height=500, width=850)

	st.plotly_chart(fig)

# Tag @ count distribution
def tag_count(data, st):

	st.subheader("Tag '@' count distribution")

	tag_list = ["@url", "@mention", "@emoji", "@hashtag"]
	sentiments = ["positive", "neutral", "negative"]

	fig = go.Figure()
	for sentiment in sentiments:
		y_val = []
		for tag in tag_list:
			tmp_data = data[data["airline_sentiment"] == sentiment]["preprocessed_text"]
			count = 0
			for dat in tmp_data:
				tokens = dat.split()
				for token in tokens:
					if tag in token:
						count += 1
			y_val.append(count)
		fig.add_trace(go.Bar(
			x=tag_list,
			y=y_val,
			name=sentiment,
		))

	fig.update_layout(barmode='group', 
						xaxis_tickangle=-60, 
						autosize=False, width=700,
						height=500)
	st.plotly_chart(fig)

# Word Cloud plot
def wordcloud(data, st, typology):
	st.header(f" WordCloud for {typology}")
	value_ll = list(set(data[typology].dropna()))
	st.subheader(f'Select for which {typology} do the WordCloud')
	v = st.radio(
			"", 
			[""] + value_ll, index = 0)
	if not v == "":
		words = ' '.join(data[data[typology] == v]["not_tag_text"].fillna(''))
		wordcloud = WordCloud(
			background_color='white',
			width=3000,
			height=2500
		).generate(words)
		fig = plt.figure(1,figsize=(12, 12))
		plt.imshow(wordcloud)
		plt.axis('off')
		st.pyplot(fig)


def to_showing_string(words):
	s = ""
	for word in words:
		s  = s + str(word[0]) + ": " + str(word[1]) + '\n'
	return s

def to_showing_string_bigram(words):
	s = ""
	for word in words:
		s  = s + word[0][0] + " " + word[0][1] + ": " + str(word[1]) + '\n'
	return s

# Frequency count
def frequency_count(data, st):
	st.header("Frequency count")
	# 5 most common words
	total_words=[]
	for lista in data["preprocessed_text"]:
		words = lista.split()
		for word in words:
			if "@" not in word:
				total_words.append(word)
	counts=Counter(total_words)
	most_occur_words = counts.most_common(5) 
	txt = to_showing_string(most_occur_words)
	html_temp = f"""
	<div style="background-color:#145796;padding:30px;border-radius:30px">
		<h1 style="color:white;text-align:center;white-space: pre-line">{txt}</h1>
	</div>
	"""
	st.subheader("1 - Top 5 most common words")
	components.html(html_temp, height = 300, width = 300)

	# 2 most common bigram
	counts = Counter()
	for sent in data["not_tag_text"]:
		words = nltk.word_tokenize(sent)
		counts.update(nltk.bigrams(words))
	most_2_bigram = counts.most_common(4) 
	txt = to_showing_string_bigram(most_2_bigram)
	html_temp = f"""
	<div style="background-color:#145796;padding:30px;border-radius:30px">
		<h1 style="color:white;text-align:center;white-space: pre-line">{txt}</h1>
	</div>
	"""
	st.subheader("2 - Top 4 most common bigram")
	components.html(html_temp, height = 300, width = 500)

	# 5 most common bigram for negative reason
	st.subheader("3 - Top 4 most common bigram by negative reason")
	negative_reasons = list(set(data["negative_reason"].dropna()))
	neg_reason = st.selectbox(
					"Select negative reason",
					[""] + negative_reasons, index = 0)
	if len(neg_reason)>0:
		counts = Counter()
		for sent in data[data["negative_reason"]==neg_reason]["not_tag_text"]:
			words = nltk.word_tokenize(sent)
			counts.update(nltk.bigrams(words))
		most_negative_common = counts.most_common(4)
		txt = to_showing_string_bigram(most_negative_common)
		html_temp = f"""
		<div style="background-color:#145796;padding:30px;border-radius:30px">
			<h1 style="color:white;text-align:center;white-space: pre-line">{txt}</h1>
		</div>
		"""
		components.html(html_temp, height = 300, width = 500)

	st.subheader("4 - Top 4 most common bigram by specified word")
	# 5 most common bigram with research word
	word_to_search = st.text_area("Insert word")
	if len(word_to_search) > 0:
		pair_freq = []
		for words in data["preprocessed_text"].fillna(""):
			word_pair_list = list(nltk.bigrams(words.split()))
			for pair in word_pair_list:
				if (pair[0] == word_to_search) or (pair[1] == word_to_search):
					if(pair[0] != pair[1]):
						if "@" not in pair[1] and "@" not in pair[0]:
							pair_freq.append(pair)
		counts=Counter(pair_freq)
		most_occur_bigram = counts.most_common(4)
		txt = to_showing_string_bigram(most_occur_bigram)
		html_temp = f"""
		<div style="background-color:#145796;padding:30px;border-radius:30px">
			<h1 style="color:white;text-align:center;white-space: pre-line">{txt}</h1>
		</div>
		"""
		components.html(html_temp, height = 300, width = 500)


# Select type of text analysis
def switch_text_analysis(analysis_code, data, st):
	if analysis_code == "1":
		word_count_distribution(data, st)
	elif analysis_code == "2":
		tag_count(data, st)
	elif analysis_code == "3":
		wordcloud(data, st, "airline_sentiment")
	elif analysis_code == "4":
		wordcloud(data, st, "negative_reason")
	elif analysis_code == "5":
		frequency_count(data, st)