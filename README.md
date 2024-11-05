## Server folder

It contains the necessary files for the Knowledge Graph creation,
the backend for the chatbot and the evaluation testing file.
In order to be able to create the knowledge graphs and proceed to use the models,
neo4j Desktop needs to be downloaded from here: https://neo4j.com/download/.

After downloading, open the app and add a local DBMS(Remember the password 
or just put 'password', as it is used later). Before starting the DBMS, click
on it, then go to Plugins and enable APOC as it is neccessary for some of its
features. After finishing the preperation start the DMBS.

  
In order to use the model an env_setup.py file needs to be created. 

The file should contain:

- OpenAI API key
- Url for the Neo4j database
- Neo4j username
- Neo4j password

For example:
os.environ["OPENAI_API_KEY"] = "sk-proj-6HmbGZK7rt1CW38KRnYsIIUUBe4isaVZQAnKCP3xm3rmasd_jer0WaMNEw9jGZ95y-R-63shljXQ6s4nY17tuCen3fYHnMjfN7lohF_gbszYA"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

After setting up the configurations, open data_ingestion.py. Change 'raw_documents'
variable to the file you want to create a knowledge graph on (The file should be
stored inside the Documents folder as a .docx).

Open chatbot.py, the backend for the chatbot UI. If you run the file (the DMBS
should be running)  the document you chose should become a knowledge graph stored
in the Neo4j database. In order to be able to see the graph, click on Open in your DMBS
and choose Neo4j Browser. When it opens type the following cypher command and run it:

MATCH (n)
RETURN (n)

You should now be able to see the knowledge graph.
![image](https://github.com/user-attachments/assets/d8167e31-c68b-429b-9a7a-6c80a0256c44)
In order to access and use the Chatbot type the command:

 npx create-react-app frontend

Replace the src/App.js file with the one inside Client/src/App.js

Install the packages that are used

Open a terminal inside the folder the frontend is stored and type the command : npm start

The chatbot should now be open in the browser.
![image](https://github.com/user-attachments/assets/ff961843-4802-4cc1-ad01-9bdb0fe84d4d)

You can now choose to use the Classic and Hybrid RAG to generate answers based on the Document you want to make questions to.
![image](https://github.com/user-attachments/assets/1cc1379c-fa86-44b6-88a3-abdf5d636073)




