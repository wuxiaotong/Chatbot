# regulaition-nlp

<h3><b>First Version:</b></h3>
Django integration with the Chatterbot

<h3><b>Second Version:</b></h3>
separate the front-end and back-end

rewrite the front-end using React

the chatbot can do calculation and greeting, and it also can 'guide the users to different webpages' according to the dunmmy tags.

<h3><b>How to run the Project:</b></h3>
1)Go into the backend directory, run python3 manage.py runserver 0.0.0.8000</br>
2)Go into the frontend directory, run npm start</br>
3)Open the browser and visit localhost:3000 to see the chatbot

<h3><b>NLP part:</b></h3>
It is located in the directory 'backend/backend/tfidfAdapter.py'. I use the tfidf model and cosine similarity to decide the result. I also add other methods to calculate the similarity but not add in the result.
