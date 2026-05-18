"""
Etapa 4: Agente ReAct con tools personalizadas
-----------------------------------------------
Usamos create_react_agent (prebuilt de LangGraph) para construir un
agente que decide cuándo invocar tools. Memoria persistente vía
MemorySaver + thread_id.
"""

import os
import math
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# --- TODO 1: Definir tools con @tool ---

@tool
def calcular(expresion: str) -> str:
    """Evalúa una expresión matemática segura (suma, resta, mult, división,
    potencias, funciones de math). Ejemplo: 'sqrt(144) + 2**3'."""
    try:
        # eval restringido: solo names del módulo math + operadores
        permitidos = {
            k: getattr(math, k) for k in dir(math) if not k.startswith("_")
        }
        return str(eval(expresion, {"__builtins__": {}}, permitidos))
    except Exception as e:
        return f"Error: {e}"


@tool
def hora_actual() -> str:
    """Devuelve la fecha y hora actuales del sistema en formato
    'YYYY-MM-DD HH:MM'."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


@tool
def clima(ciudad: str) -> str:
    """Devuelve el clima actual de una ciudad española (datos simulados).
    Ciudades soportadas: Madrid, Barcelona, Bilbao, Sevilla, Valencia."""
    datos = {
        "madrid":    "Madrid: 22°C, soleado, viento flojo del oeste.",
        "barcelona": "Barcelona: 24°C, parcialmente nublado, brisa marina.",
        "bilbao":    "Bilbao: 17°C, lluvia ligera, humedad alta (85%).",
        "sevilla":   "Sevilla: 30°C, despejado, calor seco.",
        "valencia":  "Valencia: 25°C, soleado con nubes altas.",
    }
    return datos.get(
        ciudad.lower().strip(),
        f"No tengo datos de clima para '{ciudad}'. "
        f"Ciudades disponibles: {', '.join(c.capitalize() for c in datos)}.",
    )


tools = [calcular, hora_actual, clima]

# --- TODO 2: Crear el agente ReAct con memoria ---
modelo = ChatOpenAI(model="gpt-4o-mini")
checkpointer = MemorySaver()

agente = create_react_agent(
    modelo,
    tools=tools,
    checkpointer=checkpointer,
)

# Visualizar el grafo del agente
print(agente.get_graph().draw_ascii())

# --- TODO 3: Conversación de prueba ---
config = {"configurable": {"thread_id": "alumno_1"}}

preguntas = [
    "¿Qué hora es y cuánto es la raíz cuadrada de 256?",
    "¿Qué tiempo hace en Madrid?",
    "¿Y en Bilbao?",  # debe inferir que sigue hablando del clima
]

for p in preguntas:
    print("\n" + "=" * 60)
    print(f"Usuario: {p}")
    respuesta = agente.invoke(
        {"messages": [{"role": "user", "content": p}]},
        config=config,
    )

    # El último mensaje es la respuesta final del agente
    print(f"\nAgente: {respuesta['messages'][-1].content}")

    print("\n--- Trazas internas ---")
    for m in respuesta["messages"]:
            m.pretty_print()
