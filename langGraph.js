/*1. **State:** When the user asks a question, LangGraph stores it in `state.messages` using `MessagesAnnotation`.
2. **START → LLM Node:** The graph starts and executes the `callLLM` node, which sends the user's question to Gemini.
3. **LLM Node:** Gemini understands the question and decides whether it needs to call a tool (add, multiply, or divide).
4. **Conditional Edge:** The conditional edge checks whether Gemini has requested a tool by checking `lastMessage.tool_calls`.
5. **If YES:** The graph moves to the `callTools` node, which executes the requested JavaScript function using `if-else`.
6. **Tools Node:** The tool's result is wrapped inside a `ToolMessage` and added to the existing state.
7. **Tools → LLM Edge:** The normal edge sends execution back to the LLM node so Gemini can read the tool's result.
8. **Loop:** Gemini can request another tool, and the same process repeats until the question is answered.
9. **If NO:** When Gemini returns a normal response without any tool calls, the conditional edge directs execution to `END`.
10. END: The graph terminates and returns the final answer to the user.

Remember:

- **Node =** A function that performs some work.
- **Edge =** A connection that decides where execution goes next.
- **Conditional Edge =** A connection that chooses the next node based on a condition.
*/
import "dotenv/config";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { tool } from "@langchain/core/tools";
import {
    StateGraph,
    MessagesAnnotation,
    START,
    END
} from "@langchain/langgraph";

import {
    HumanMessage,
    ToolMessage
} from "@langchain/core/messages";


// 1. DEFINE TOOLS

const add = tool(
    ({ a, b }) => a + b,
    {
        name: "add",
        description: "Add two numbers",
        schema: {
            type: "object",
            properties: {
                a: { type: "number" },
                b: { type: "number" }
            },
            required: ["a", "b"]
        }
    }
);

const multiply = tool(
    ({ a, b }) => a * b,
    {
        name: "multiply",
        description: "Multiply two numbers",
        schema: {
            type: "object",
            properties: {
                a: { type: "number" },
                b: { type: "number" }
            },
            required: ["a", "b"]
        }
    }
);

const divide = tool(
    ({ a, b }) => {
        if (b === 0) {
            throw new Error("Cannot divide by zero");
        }
        return a / b;
    },
    {
        name: "divide",
        description: "Divide the first number by the second number",
        schema: {
            type: "object",
            properties: {
                a: { type: "number" },
                b: { type: "number" }
            },
            required: ["a", "b"]
        }
    }
);


// 2. INITIALIZE GEMINI

const tools = [add, multiply, divide];

const llm = new ChatGoogleGenerativeAI({
    model: "gemini-3.5-flash-lite",
    temperature: 0
});
// integrate tools 
const llmWithTools = llm.bindTools(tools);


// 3. LLM NODE

async function callLLM(state) {
    console.log("before llm node",state);
    const response = await llmWithTools.invoke(state.messages);
    
    return {
        messages: [response]
    };
}


// 4. TOOLS NODE

async function callTools(state) {
    console.log("before tool node",state);
    const lastMessage =
        state.messages[state.messages.length - 1];

    const toolCalls = lastMessage.tool_calls;

    const results = [];

    for (const toolCall of toolCalls) {

        const { name, args, id } = toolCall;

        let result;

        if (name === "add") {
            result = await add.invoke(args);
        }

        else if (name === "multiply") {
            result = await multiply.invoke(args);
        }

        else if (name === "divide") {
            result = await divide.invoke(args);
        }

        else {
            throw new Error(`Unknown tool: ${name}`);
        }

        results.push(
            new ToolMessage({
                content: String(result),
                tool_call_id: id
            })
        );
    }

    return {
        messages: results
    };
}


// 5. BUILD GRAPH

const graph = new StateGraph(MessagesAnnotation)

    .addNode("llm", callLLM)
    .addNode("tools", callTools)

    .addEdge(START, "llm")
    // after llm we can either call tool or stop, that will be decided by llm
    .addConditionalEdges("llm", (state) => {
        // get last mssg (send by llm)
        const lastMessage =
            state.messages[state.messages.length - 1];

        // need tool calling
        if (lastMessage.tool_calls?.length > 0) {
            return "tools";
        }
        // stop it
        else {
            return END;
        }

    })

    .addEdge("tools", "llm")
    .compile();


// 6. INVOKE GRAPH

const result = await graph.invoke({

    messages: [
        new HumanMessage(
            "What is 20 multiplied by 5 and then divided by 2?"
        )
    ]

});


// 7. FINAL OUTPUT

console.log("final ans======",
    result.messages[result.messages.length - 1].content
);



/*
{
    messages: [
        HumanMessage("What is 10 multiplied by 5?"),

        AIMessage({
            tool_calls: [
                {
                    name: "multiply",
                    args: { a: 10, b: 5 },
                    id: "call_123"
                }
            ]
        })
    ]
}


{
    messages: [
        HumanMessage(...),
        AIMessage(...),

        ToolMessage({
            content: "50",
            tool_call_id: "call_123"
        })
    ]
}
*/
