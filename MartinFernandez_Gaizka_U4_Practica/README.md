# Unidad 3 · Sesión 2 — LangGraph

Cuatro etapas incrementales construyendo un asistente con LangGraph:

| Archivo | Qué añade |
|---|---|
| `etapa1_chatbot.py` | `StateGraph` mínimo con un único nodo. |
| `etapa2_memoria.py` | `MemorySaver` + `thread_id` para conversación persistente. |
| `etapa3_router.py` | Estado personalizado y `add_conditional_edges` con clasificador. |
| `etapa4_agente.py` | Agente ReAct con `create_react_agent` y 3 tools (`@tool`). |

---

## 1. ¿En qué etapa se ve más claramente la ventaja de LangGraph sobre una Chain LCEL?

En la **Etapa 3 (router)** y, sobre todo, en la **Etapa 4 (agente ReAct)**.

Una Chain LCEL (`prompt | modelo | parser`) es esencialmente un *pipeline* lineal: los datos fluyen en una sola dirección y cada paso sabe a quién pasa el resultado en tiempo de definición. La Etapa 1 podría reescribirse en LCEL en menos líneas, y la Etapa 2 se puede emular con `RunnableWithMessageHistory`, así que ahí LangGraph todavía parece "ceremonia extra".

La cosa cambia cuando aparecen **bifurcaciones decididas en runtime** (Etapa 3) y, sobre todo, **bucles** (Etapa 4): el agente ReAct entra al nodo del modelo, decide si llamar a una tool, vuelve al modelo con el resultado, posiblemente llama a otra tool, y así hasta que considere que tiene la respuesta. Eso es un *while* sobre el grafo, no una pipeline. Reproducirlo en LCEL exige construir el bucle a mano (con `RunnableLambda`, condicionales, gestión de mensajes intermedios, parada por número de iteraciones…), justo lo que `create_react_agent` te da gratis en cinco líneas.

## 2. Bug / malentendido más difícil de depurar

Confundir cómo se **fusiona el estado** entre nodos. En la Etapa 1 mi primera versión devolvía `{"messages": respuesta}` (la `AIMessage` directamente, no en lista). El reducer `add_messages` de `MessagesState` espera siempre una lista de mensajes para concatenarlos al historial; si pasas un objeto suelto, o bien explota con un `TypeError` poco claro, o bien lo trata como si fuera un único mensaje y se rompe en cuanto otro nodo (por ejemplo, en la Etapa 3) intenta hacer `state["messages"][-1].content`.

La regla mental que me cuesta interiorizar: el `dict` que devuelve un nodo **no reemplaza** el estado, sino que cada clave se mezcla con la existente usando su reducer. Por eso en la Etapa 3, el nodo `clasificar` puede devolver solo `{"categoria": cat}` sin tocar `messages` y todo sigue funcionando: las demás claves quedan intactas.

## 3. Diseño de un grafo para una aplicación real

**Aplicación: triaje de tickets de soporte para una pequeña SaaS.**

- **Estado:**
  - `messages`: historial de la conversación con el cliente (con `add_messages`).
  - `cliente_id`: id del usuario que abre el ticket.
  - `prioridad`: `low` / `medium` / `high` (la rellena el clasificador).
  - `categoria`: `billing` / `bug` / `how_to` / `account`.
  - `resuelto`: bool — bandera para terminar el bucle.

- **Nodos:**
  - `clasificar_ticket` → llena `categoria` y `prioridad`.
  - `responder_faq` → intenta resolver con RAG sobre la base de conocimiento.
  - `escalar_humano` → crea un issue en Zendesk y notifica al equipo.
  - `consultar_billing` → tool que lee el estado de la suscripción del cliente.
  - `verificar_resolucion` → pregunta al cliente si su problema está resuelto.

- **Aristas:**
  - `START → clasificar_ticket`.
  - Edge condicional desde `clasificar_ticket`: si `prioridad == "high"` o `categoria == "bug"` → `escalar_humano → END`; si `categoria == "billing"` → `consultar_billing → responder_faq`; en otro caso → `responder_faq`.
  - `responder_faq → verificar_resolucion`.
  - Edge condicional desde `verificar_resolucion`: si `resuelto` → `END`; si no → `escalar_humano → END`.

El estado persiste con `SqliteSaver` usando `thread_id = ticket_id`, así un cliente puede retomar la conversación días después.
