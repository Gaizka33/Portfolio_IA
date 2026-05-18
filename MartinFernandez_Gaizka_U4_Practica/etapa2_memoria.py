"""
Etapa 2: Chatbot con memoria (MemorySaver)
-------------------------------------------
Añadimos memoria persistente al chatbot mediante un checkpointer
y conversamos por turnos con un thread_id estable.
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

load_dotenv()

modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def llamar_modelo(state: MessagesState) -> dict:
    return {"messages": [modelo.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("chatbot", llamar_modelo)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# TODO 1: Crear el MemorySaver y compilar con checkpointer
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# TODO 2: Configuración con thread_id
# El thread_id identifica una conversación concreta; el checkpointer
# guarda el estado bajo esa clave y lo recupera en cada invocación.
config = {"configurable": {"thread_id": "alumno_1"}}

# TODO 3: Bucle interactivo
print("Chatbot con memoria. Escribe 'salir' para terminar.\n")
while True:
    user_msg = input("Tú: ").strip()
    if user_msg.lower() in {"salir", "exit", "quit"}:
        break
    if not user_msg:
        continue

    resultado = graph.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config=config,
    )
    print(f"Bot: {resultado['messages'][-1].content}\n")

# TODO 4: Mostrar cuántos mensajes hay guardados en el thread
estado = graph.get_state(config)
print(f"\nMensajes guardados en el thread '{config['configurable']['thread_id']}': "
      f"{len(estado.values['messages'])}")


# --------------------------------------------------------------
# Reto adicional (opcional): persistencia en SQLite
# --------------------------------------------------------------
# Para que el historial sobreviva entre ejecuciones del script,
# sustituye las líneas del checkpointer por:
#
#     from langgraph.checkpoint.sqlite import SqliteSaver
#     checkpointer = SqliteSaver.from_conn_string("chat.db")
#
# (Requiere: pip install langgraph-checkpoint-sqlite)
#
# Con MemorySaver el historial vive solo en RAM: al matar el proceso,
# se pierde. Con SqliteSaver el estado se serializa a un fichero
# .db y la próxima ejecución lo lee de disco, así que el bot sigue
# "recordando" lo que dijiste en sesiones anteriores con el mismo
# thread_id.
# --------------------------------------------------------------
