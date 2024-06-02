import warnings
import os
import json
import pickle
from tqdm import tqdm
import numpy as np
from langchain_community.document_loaders import JSONLoader
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Tuple
from langchain_experimental.text_splitter import SemanticChunker
from transformers import Pipeline

warnings.filterwarnings('ignore')


class ChatBotRAG:
    def __init__(self, data_path: str, vector_db_path: str, embeddings_model: str, llm_name: str, light_llm_name: str):
        self.data_path = data_path
        self.vector_db_path = vector_db_path
        self.embeddings_model = embeddings_model
        self.llm_name = llm_name
        self.light_llm_name = light_llm_name

        self.documents_raw_dict = {}
        self.documents = []
        self.embeddings = OllamaEmbeddings(model=embeddings_model)
        self.knowledge_vector_db = None

    def load_data(self):
        with open(self.data_path, 'r') as f:
            data = json.loads(f.read())
        
        for i in range(len(data['data'])):
            data['data'][i]['id_doc'] = i
            self.documents_raw_dict[data['data'][i]['id_doc']] = data['data'][i]['description']
        
        with open('dataset.json', 'w') as fout:
            json.dump(data['data'], fout)
        
        loader = JSONLoader(
            file_path='./dataset.json',
            jq_schema='.[]',
            content_key='description',
            metadata_func=self.metadata_func
        )
        
        self.documents = loader.load()[:]
        for doc in self.documents:
            doc.page_content = doc.metadata['title'] + ' ' + doc.page_content
    
    @staticmethod
    def metadata_func(content: dict, metadata: dict) -> dict:
        metadata['id_doc'] = content.get('id_doc')
        metadata["title"] = content.get("title")
        metadata["parent_title"] = content.get("parent_title")
        metadata["business_line_id"] = content.get("business_line_id")
        metadata["direction"] = content.get("direction")
        metadata["product"] = content.get("product")
        metadata["type"] = content.get("type")
        metadata["url"] = content.get("url")
        return metadata

    def split_documents(self) -> List[LangchainDocument]:

        if os.path.exists('docs_processed_unique'):
            with open('docs_processed_unique', 'rb') as docs_file:
                docs_processed_unique = pickle.load(docs_file)
        
        else:
            text_splitter = SemanticChunker(self.embeddings, breakpoint_threshold_type="percentile")
            docs_processed = []
            
            for doc in tqdm(self.documents):
                docs_processed += text_splitter.split_documents([doc])
            
            unique_texts = {}
            docs_processed_unique = []
            for doc in docs_processed:
                if doc.page_content not in unique_texts:
                    unique_texts[doc.page_content] = True
                    docs_processed_unique.append(doc)
    
            with open('docs_processed_unique', 'wb') as docs_file:
                pickle.dump(docs_processed_unique, docs_file)
        
        return docs_processed_unique

    def create_vector_database(self, docs_processed: List[LangchainDocument]):
        print(self.vector_db_path)
        print(os.path.exists(self.vector_db_path))
        if not os.path.exists(self.vector_db_path):
            self.knowledge_vector_db = Qdrant.from_documents(
                docs_processed,
                self.embeddings,
                path=self.vector_db_path,
                collection_name="my_documents",
            )
        else:
            self.knowledge_vector_db = Qdrant.from_existing_collection(self.embeddings, path=self.vector_db_path, collection_name="my_documents")

    def answer_with_rag(
        self,
        question: str,
        num_retrieved_docs: int = 5,
        num_docs_final: int = 5
    ) -> Tuple[str, List[LangchainDocument]]:
        questions = self.fusion_questions(question, self.light_llm_name, 4)
        relevant, scores = [], []
        
        for question in questions:
            rels = self.knowledge_vector_db.similarity_search_with_score(query=question, k=num_retrieved_docs)
            for rel in rels:
                relevant.append(rel[0])
                scores.append(rel[1])
        
        unique_texts = {}
        relevant_unique, scores_unique = [], []
        for i, doc in enumerate(relevant):
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                relevant_unique.append(doc)
                scores_unique.append(scores[i])
        
        relevant_docs = [doc for i, doc in enumerate(relevant_unique) if scores_unique[i] > np.quantile(scores_unique, 0.9)]
        
        unique_urls = {}
        relevant_unique_ = []
        for doc in relevant_docs:
            if doc.metadata['url'] not in unique_urls:
                unique_urls[doc.metadata['url']] = True
                relevant_unique_.append(doc)
        
        checked_relevant_unique_ = self.check_relevance(questions, relevant_unique_, scores_unique)

        checked_relevant_unique_ = checked_relevant_unique_[:num_docs_final]
        
        relevant_docs = [self.documents_raw_dict[x.metadata['id_doc']] for x in checked_relevant_unique_]
        
        llm = ChatOllama(model=self.llm_name, temperature=0)
        context = "\nИзвлеченные документы:\n" + "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])
        
        messages = [
            SystemMessage(content="Цитируя и используя информацию из контекста ответь на вопрос. Если ответ не может быть выведен из контекста или контекст пуст, не отвечай. Отвечай кратко без лишней информации."),
            HumanMessage(content=f"Контекст: {context} --- Вопрос, на который нужно ответить: {question}"),
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({})
        
        return answer, [x.metadata['url'] for x in checked_relevant_unique_]
    
    def fusion_questions(self, question: str, llm_name: Pipeline, num_questions: int = 2) -> List[str]:
        llm = ChatOllama(model=llm_name, temperature=0)
        questions = [question]
        
        for _ in range(num_questions):
            messages = [
                SystemMessage(content="Ты интеллектуальный помощник, улучшающий чатбот техподдержки. Твоя задача генирировать альтернативные формулировки вопроса пользователя. Пиши только сгенерированный запрос, без другого текста."),
                HumanMessage(content=f"Вопрос, который нужно переформулировать: {question}")
            ]
            
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({})
            
            questions.append(answer)
        
        return questions

    def check_relevance(self, questions: List[str], relevant_docs: List[LangchainDocument], scores_unique) -> List[LangchainDocument]:
        checked_relevant_unique_ = []
        
        for doc in tqdm(relevant_docs):
            yes_q, no_q = 0, 0
            for quest in questions:
                doc_content = self.documents_raw_dict[doc.metadata['id_doc']]
                messages = [
                    SystemMessage(content="Твоя задача проверить ответ на соответсвие вопросу. Отвечай только да или нет"),
                    HumanMessage(content=f"Вопрос: {quest} --- Ответ: {doc_content}"),
                ]
                
                prompt = ChatPromptTemplate.from_messages(messages)
                chain = prompt | ChatOllama(model=self.light_llm_name, temperature=0) | StrOutputParser()
                answer = chain.invoke({})
                
                if answer.strip().lower() == 'да':
                    yes_q += 1
                else:
                    no_q += 1
            
            if yes_q > no_q or (yes_q > 0 and max(scores_unique) > 0.82) or max(scores_unique) > 0.84:
                checked_relevant_unique_.append(doc)
        
        return checked_relevant_unique_