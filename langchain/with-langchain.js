import dotenv from "dotenv";
dotenv.config({ path: "../.env" });

import {ChatGoogleGenerativeAI} from "@langchain/google-genai";
import { createAgent, tool } from "langchain";
import readlineSync from "readline-sync";

// GEMINI MODEL 

const model = new ChatGoogleGenerativeAI({
    model: "gemini-3.5-flash-lite",
    apiKey: process.env.GEMINI_API_KEY
});


//  TOOL 1: WEATHER

const weatherinfo = tool(
    async ({ city }) => {

        const res = await fetch(
            `https://wttr.in/${encodeURIComponent(city)}?format=j1`
        );
        if (!res.ok) {
            throw new Error("Weather API failed");
        }
        const data = await res.json();
        return {
            city,
            temperature: data.current_condition[0].temp_C
        };
    },
    {
        name: "weatherinfo",
        description: "Get the current weather of a city",
        schema: {
            type: "object",
            properties: {
                city: {
                    type: "string",
                    description: "Name of the city"
                }
            },
            required: ["city"],
            additionalProperties: false
        }
    }
);


// ---------------- TOOL 2: COIN INFO ----------------

const coininfo = tool(
    async ({ coin }) => {
        const res = await fetch(
            `https://api.coingecko.com/api/v3/simple/price?ids=${encodeURIComponent(coin)}&vs_currencies=inr`
        );

        if (!res.ok) {
            throw new Error("CoinGecko API failed");
        }

        const data = await res.json();
        return data;
    },
    {
        name: "coininfo",
        description: "Get the current price of a cryptocurrency in INR",
        // JSON Schema instead of Zod
        schema: {
            type: "object",
            properties: {
                coin: {
                    type: "string",
                    description: "Coin ID, eg. bitcoin or solana"
                }
            },
            required: ["coin"],
            additionalProperties: false
        }
    }
);


// ---------------- CREATE AGENT ----------------

const agent = createAgent({
    model,
    tools: [weatherinfo, coininfo],

    systemPrompt: `
        You can use weatherinfo to get weather information.
        You can use coininfo to get cryptocurrency prices.
    `
});
// benefit of langchain is we dont have to loop through all functionCalls and get answer and manually push it in history , langchain abstracts all these steps and gives simple codelines
// RUN AGENT- 
async function main(query) {
    const result = await agent.invoke({
        messages: [
            {
                role: "user",
                content: query
            }
        ]
    });

    const finalMessage = result.messages.at(-1);
    console.log("\nAI:", finalMessage.content);
}


// USER INPUT 

while (true) {
    const query = readlineSync.question("\nYou: ");
    if (query.trim().toLowerCase() === "exit") {
        break;
    }
    try {
        await main(query);
    } catch (error) {
        console.error("Error:", error.message);
    }
}
