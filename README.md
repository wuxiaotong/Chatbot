# regulaition-nlp

<h3><b>First Version:</b></h3>
Django integration with the Chatterbot

<h3><b>Second Version:</b></h3>
separate the front-end and back-end

rewrite the front-end using React

the chatbot can do calculation and greeting, and it also can 'guide the users to different webpages' according to the dunmmy tags.
<h3><b>NLP part:</b></h3>
It is located in the directory 'backend/backend/jupyter_model/tfidfAdapter.py'. I use the tfidf model and cosine similarity to decide the result. I also add other methods to calculate the similarity but not add in the result.


<h3><b>Third Version:</b></h3>
Add different models in the directory 'backend/backend/jupyter-model/'.</br>
Put all the adapter in the directory 'backend/backend/adapter'</br>
The follows are the description of the models in the directory 'backend/backend/jupyter-model/':</br>
1.'document_similarity': build different models including LDA, TFIDF, LSI, DOC2VEC, DIFFLIB for compare the document similarity.</br>
2.'baseline_model': a simple classification model for detecting whther the questions and answers are matched.</br>
3.'test_cnn_model': a cnn model for ranking the answers.</br>
4.'similarity_model': a feature-based model for detecting whether the questions and answers are matched. The model is also used to rank the answers.

<h3><b>How to run the Project:</b></h3>
1)Go into the backend directory, run python3 manage.py runserver 0.0.0.8000</br>
2)Go into the frontend directory, run npm start</br>
3)Open the browser and visit localhost:3000 to see the chatbot


<h3><b>See all the models:</b></h3>
1)Go into the 'backend/backend/jupyter_model' directory, using jupyter notebook to see the models</br>
