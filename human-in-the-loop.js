import "dotenv/config";
import { GoogleGenAI } from "@google/genai";
import readlineSync from "readline-sync"

const ai=new GoogleGenAI({
	apiKey:process.env.GEMINI_API_KEY
})

// fn-1
async function coininfo({coin}){
	const res=await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${coin}&vs_currencies=inr`);
	const data=await res.json();
	return data;
}

// fn-2
async function weatherinfo({city}) {
	const res=await fetch(`https://wttr.in/${encodeURIComponent(city)}?format=j1`);

	const data=await res.json();
	return {
		temperature:data.current_condition[0].temp_C
	};
}
// all tool definition

const toolsdesc=[
	{
		functionDeclarations:[
			{
				name:"weatherinfo",
				description:"get the current weather of a city eg.delhi",
				parameters:{
					type:"OBJECT",
					properties:{
						city:{
							type:"STRING",
							description:"name of city"
						}
					},
					required:["city"]
				}
			},
			{
				name:"coininfo",
				description:"get the current price of a coin eg.solana",
				parameters:{
					type:"OBJECT",
					properties:{
						coin:{
							type:"STRING",
							description:"name of coin"
						}
					},
					required:["coin"]
				}
			}
		]
	}
];



const tools={
	weatherinfo:weatherinfo,
	coininfo:coininfo
};

async function main(query) {
	let history=[
		{
			role:"user",
			parts:[{text:query}]
		}
	];
	while(true){
		const response=await ai.models.generateContent({
			model:"gemini-3.5-flash-lite",
			contents:history,
			config:{
				tools:toolsdesc,
			}
		});
		// no tool call needed by llm
		if(!response.functionCalls|| response.functionCalls.length===0){
			console.log("AI:",response.text);
			break;
		}
		// else call tools
		console.log("calling:",response.functionCalls);
		history.push(response.candidates[0].content);

		const responseParts=[];

		for(const functionCall of response.functionCalls){
			let result;

			// HUMAN IN THE LOOP
			if(functionCall.name==="coininfo"){
				const answer=readlineSync.question(
				`AI wants to call coininfo for ${functionCall.args.coin}. Allow? (yes/no):`
				)

				if(answer.trim().toLowerCase()==="yes"){
					console.log("coin api tool approved");

					result=await tools[functionCall.name](functionCall.args);
				}//rejected
				else{
					console.log("coin api tool rejected");
					result={error:"user denied permisiion"}
				}
			}
			else if(functionCall.name==="weatherinfo"){
				result=await tools[functionCall.name](functionCall.args);
			}
			else{
				result={error:"unkown tool"}
			}

			// storing all the results of every tool calls
			 responseParts.push({
                functionResponse: {
                    name: functionCall.name,
                    response: result
                }
            });

		}
		 history.push({
            role: "user",
            parts: responseParts
        });
	}
}


while(true){
	const query=readlineSync.question("You: ");
	if(query.trim().toLowerCase()==="exit")
		break;
	await main(query);
}
