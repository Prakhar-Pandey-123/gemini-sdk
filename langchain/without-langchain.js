import "dotenv/config";
import { GoogleGenAI } from "@google/genai";
import readlineSync from "readline-sync";

const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY
});

const model = "gemini-2.5-flash";

// TOOLS 

async function weatherinfo({ city }) {
    const res = await fetch(
        `https://wttr.in/${encodeURIComponent(city)}?format=j1`
    );

    const data = await res.json();
    return {
        city,
        temperature: data.current_condition[0].temp_C
    };
}

async function coininfo({ coin }) {
    const res = await fetch(
        `https://api.coingecko.com/api/v3/simple/price?ids=${encodeURIComponent(coin)}&vs_currencies=inr`
    );

    return await res.json();
}

// TOOL DECLARATIONS 

const toolsdesc = [
    {
        functionDeclarations: [
            {
                name: "weatherinfo",
                description: "Get current weather of a city",
                parameters: {
                    type: "OBJECT",
                    properties: {
                        city: { type: "STRING" }
                    },
                    required: ["city"]
                }
            },
            {
                name: "coininfo",
                description: "Get current cryptocurrency price",
                parameters: {
                    type: "OBJECT",
                    properties: {
                        coin: { type: "STRING" }
                    },
                    required: ["coin"]
                }
            }
        ]
    }
];

// Map tool names to actual JavaScript functions
const tools = {
    weatherinfo,
    coininfo
};

//  AGENT LOOP 

async function runAgent(query) {

    const history = [
        {
            role: "user",
            parts: [{ text: query }]
        }
    ];

    // Prevent an infinite tool-calling loop
    const MAX_ITERATIONS = 10;

    for (let i = 0; i < MAX_ITERATIONS; i++) {

        // STEP 1: Ask LLM what to do next
        const response = await ai.models.generateContent({
            model,
            contents: history,
            config: {
                tools: toolsdesc
            }
        });

        // STEP 2: Check whether LLM wants tools
        const functionCalls = response.functionCalls || [];

        // No tools requested = final answer
        if (functionCalls.length === 0) {
            console.log("\nAI:", response.text);
            return;
        }

        // STEP 3: Save LLM's tool-call message
        history.push(response.candidates[0].content);

        const responseParts = [];

        // STEP 4: Execute all requested tools
        for (const call of functionCalls) {

            const tool = tools[call.name];
            let result;

            if (!tool) {
                result = { error: "Unknown tool" };
            } else {
                try {
                    result = await tool(call.args);
                } catch (error) {
                    result = { error: error.message };
                }
            }

            // STEP 5: Prepare tool result for LLM
            responseParts.push({
                functionResponse: {
                    name: call.name,
                    response: {
                        result
                    }
                }
            });
        }

        // STEP 6: Send all tool results back together
        history.push({
            role: "user",
            parts: responseParts
        });

        // LOOP repeats:
        // LLM receives the updated history,
        // thinks again and decides the next action.
    }

    console.log("Maximum iterations reached.");
}

//  USER INPUT 

while (true) {

    const query = readlineSync.question("\nYou: ");

    if (query.trim().toLowerCase() === "exit") {
        break;
    }

    await runAgent(query);
}
