# SalesmanChatbot
A chatbot that mimics a Salesman for a Cleaning company

### Project
Tasked with developing a chatbot for a German-based website that automates sales and 
customer interactions. The bot will need to handle inquiries about the company’s products 
and guide users toward purchasing the appropriate product package.

### Implementation
In this implementation, a Retrieval-Augmented Generation (RAG) approach is adopted 
to create a highly responsive and accurate chatbot system. This technique integrates 
retrieval-based mechanisms with generative models to enhance the chatbot’s performance
and contextual understanding.

### Why RAG?
Retrieval-Augmented Generation (RAG) combines retrieval-based and generative models
to enhance response accuracy and relevance. By retrieving relevant documents from a 
predefined corpus, RAG ensures the chatbot generates both fluent and factually correct 
answers. This approach is ideal for tasks like product recommendations and sales 
automation, where reliable and accurate information is essential.

### Approach
The data was manually gathered and stored in a text file, which was read using
Langchain’s TextLoader. This initial step allowed for easy handling of the text data. To
manage the large volume of text, the RecursiveCharacterTextSplitter was used to divide
the document into smaller chunks. This approach ensures that the document is split into
meaningful pieces, facilitating efficient processing in subsequent stages.

For embedding the text, the Huggingface model all-MiniLM-L6-v2 was employed. The
embeddings capture the semantic information of each chunk, making it easier for the model
to retrieve relevant content when queried. These embeddings were stored in a vector store,
providing a persistent and organized method for efficient retrieval.

The chat prompt defines a professional chatbot for Hygienewunder, a hygiene product
company. The bot provides detailed product knowledge, recommends packages based on user
input (like allergies or pet ownership), and guides users through the sales process without
offering any discounts. It references official product details for accuracy and ensures responses are clear, polite, and formal in German when needed. The chatbot engages users with
clarifying questions to tailor recommendations, focusing on providing helpful and accurate
information for a seamless customer experience.

A low temperature value of 0.4 was used to design the chatbot to ensure that its responses
are controlled, focused, and consistent. This minimizes randomness in generated answers,
leading to more reliable and accurate responses, especially when handling product inquiries
and sales interactions. Such precision is crucial for maintaining a professional tone and
delivering fact-based information.

### Conclusion
In conclusion, the development of the Hygienewunder chatbot represents a
significant advancement in automating customer interactions for a hygiene and cleaning
products company. By employing a Retrieval-Augmented Generation (RAG) approach, the
chatbot effectively addresses customer inquiries and provides tailored product 
recommendations. While the system is currently functional and provides accurate 
information, ongoing work focuses on integrating a secure payment solution to facilitate 
seamless transactions. This enhancement will further streamline the sales process, making it more efficient and user-friendly for customers.
