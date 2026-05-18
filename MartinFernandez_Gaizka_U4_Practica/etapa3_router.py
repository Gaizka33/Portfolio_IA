"""
Etapa 3: Enrutado condicional - Asistente con especialidades
-------------------------------------------------------------
Un clasificador decide a qué experto (python / sql / general) enviar
la pregunta del usuario mediante un edge condicional.

    START → clasificar ──┬─► experto_python ──┐
                         ├─► experto_sql ─────┼─► END
                         └─► experto_general ─┘
"""

import os
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

load_dotenv()

modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# TODO 1: Estado personalizado con messages + categoria
class EstadoRouter(TypedDict):
    messages: Annotated[list, add_messages]
    categoria: str


# --- Nodo clasificador ---
def clasificar(state: EstadoRouter) -> dict:
    pregunta = state["messages"][-1].content
    prompt = (
        "Clasifica la siguiente pregunta en UNA SOLA palabra entre: "
        "'python', 'sql' o 'general'. Responde SOLO con esa palabra.\n\n"
        f"Pregunta: {pregunta}"
    )
    salida = modelo.invoke(prompt).content.strip().lower()

    # Normalizar la salida (puede contener basura)
    if "python" in salida:
        cat = "python"
    elif "sql" in salida:
        cat = "sql"
    else:
        cat = "general"

    return {"categoria": cat}


# --- TODO 2: Nodos especialistas ---
def experto_python(state: EstadoRouter) -> dict:
    system = "Eres un experto en Python. Responde con código limpio y comentarios."
    pregunta = state["messages"][-1].content
    respuesta = modelo.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": pregunta},
    ])
    return {"messages": [respuesta]}


def experto_sql(state: EstadoRouter) -> dict:
    system = (
        "Eres un experto en SQL. Responde con sentencias SQL claras "
        "(SELECT, JOIN, GROUP BY, etc.), comentadas si hace falta, "
        "y explica brevemente lo que hace la consulta."
    )
    pregunta = state["messages"][-1].content
    respuesta = modelo.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": pregunta},
    ])
    return {"messages": [respuesta]}


def experto_general(state: EstadoRouter) -> dict:
    system = (
        "Eres un asistente generalista, claro y conciso. "
        "Responde en español y sin código a menos que sea imprescindible."
    )
    pregunta = state["messages"][-1].content
    respuesta = modelo.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": pregunta},
    ])
    return {"messages": [respuesta]}


# --- TODO 3: Función router ---
def decidir_ruta(
    state: EstadoRouter,
) -> Literal["experto_python", "experto_sql", "experto_general"]:
    """Devuelve el nombre del nodo siguiente según state['categoria']."""
    cat = state["categoria"]
    if cat == "python":
        return "experto_python"
    if cat == "sql":
        return "experto_sql"
    # Fallback de seguridad: cualquier categoría inesperada cae aquí
    return "experto_general"


# --- Construir el grafo ---
builder = StateGraph(EstadoRouter)
builder.add_node("clasificar", clasificar)
builder.add_node("experto_python", experto_python)
builder.add_node("experto_sql", experto_sql)
builder.add_node("experto_general", experto_general)

builder.add_edge(START, "clasificar")

# TODO 4: Edge condicional desde "clasificar"
builder.add_conditional_edges("clasificar", decidir_ruta)

# TODO 5: Cada experto conecta a END
builder.add_edge("experto_python", END)
builder.add_edge("experto_sql", END)
builder.add_edge("experto_general", END)

graph = builder.compile()

# Visualizar el grafo
print(graph.get_graph().draw_ascii())

# --- Probar con tres preguntas distintas ---
preguntas = [
    "¿Cómo escribo una list comprehension?",
    "¿Cómo hago un LEFT JOIN entre dos tablas?",
    "¿Cuál es la capital de Francia?",
]

for p in preguntas:
    print("\n" + "=" * 60)
    print(f"Pregunta: {p}")
    resultado = graph.invoke({"messages": [{"role": "user", "content": p}]})
    print(f"Categoría detectada: {resultado['categoria']}")
    print(f"Respuesta: {resultado['messages'][-1].content[:200]}...")
