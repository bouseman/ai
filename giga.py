from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
import hvac
import json



def get_giga_auth():

    VAULT_ADDR = "https://vault.gipnotik.ru" 
    ROLE_ID = os.getenv('VAULT_ROLE_ID')
    SECRET_ID = os.getenv('VAULT_SECRET_ID')

    client = hvac.Client(url=VAULT_ADDR)

    client.auth.approle.login(
        role_id=ROLE_ID,
        secret_id=SECRET_ID
    )
        
    giga_key = client.secrets.kv.v2.read_secret_version(path='gigachat', mount_point='kv', raise_on_deleted_version=True)['data']['data']['authorization_key']

    return giga_key



giga = GigaChat(
    credentials=get_giga_auth(),
    verify_ssl_certs=False,
)

#print(giga.invoke("Hello, world!"))

embeddings = GigaChatEmbeddings(credentials=get_giga_auth(), verify_ssl_certs=False)
#result = embeddings.embed_documents(texts=["Привет!"])
#print(result)

# 1. Загрузка PDF документа
pdf_path = "Исаев резюме.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()


# 2. Разделение текста на чанки
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # размер чанка
    chunk_overlap=200,  # перекрытие между чанками
    length_function=len,
    separators=["\n\n\n", "\n\n", "\n"]    
)

chunks = text_splitter.split_documents(documents)
print(f"Создано чанков: {len(chunks)}")

# 4. Создание и сохранение векторной базы
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db",
)


# 3. Создаем RAG цепочку
qa_chain = RetrievalQA.from_chain_type(
    llm=giga,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# 4. Задаем вопросы
while True:
    question = input("Ваш вопрос: ")
    if question.lower() == 'выход':
        break
    
    result = qa_chain.invoke({"query": question})
    
    print(f"\nОтвет: {result['result']}")
    
    # Показать источники
    for i, doc in enumerate(result['source_documents']):
        print(f"\nИсточник {i+1}: {doc.page_content[:150]}...")
    
    print("\n" + "="*50)

