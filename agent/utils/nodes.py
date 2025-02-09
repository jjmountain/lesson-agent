from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing import Literal
from agent.utils.state import State
from datetime import datetime

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        model =  ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    return model

llm = _get_model("openai")

# Pydantic class defining the structured output for the router
class TeachingRoute(BaseModel):
    step: Literal[
        "give_context",
        "present_word",
        "check_meaning",
        "give_hint",
        "model_pronunciation",
        "evaluate_pronunciation",
        "next_word",
        "redirect", # handle off-track interactions
        "end_lesson", # end disruptive lessons
        "end"
    ] = Field(
        description="The next step in the Japanese langauge lesson teaching process",
        default="end"
    )
    response_text: str = Field(
        description="The text to show to the user"
    )
    is_disruptive: bool = Field(
        description="Whether the user's behavior is disruptive"
    )

# Augment LLM with structured output
router = _get_model("openai").with_structured_output(TeachingRoute)

# Router node
def teaching_router(state: State):
    """Route with disruption handling in state"""
    decision = router.invoke(
        [
            SystemMessage(content="""You are a Japanese language teacher deciding the next step in a vocabulary lesson.
            Monitor for disruptive behavior while maintaining professionalism.
            
            Examples of different teaching situations and responses:
            
            1. Initial Context:
            Input: (lesson start)
            Response: {
                "step": "give_context",
                "response_text": "今日は便利な電化製品について勉強しましょう。",
                "is_disruptive": false
            }
            
            2. Presenting Word:
            Input: (after context)
            Response: {
                "step": "present_word",
                "response_text": "この文を聞いてください：食洗器で皿を洗います。",
                "is_disruptive": false
            }
            
            3. Checking Meaning (good attempt):
            Input: "Is it something for washing dishes?"
            Response: {
                "step": "model_pronunciation",
                "response_text": "はい、その通りです！食洗器は英語でdishwasherですね。では、発音を練習しましょう。",
                "is_disruptive": false
            }
            
            4. Checking Meaning (needs hint):
            Input: "Is it a washing machine?"
            Response: {
                "step": "give_hint",
                "response_text": "惜しいですね。台所で使う機械です。特に食器を洗うための機械ですよ。",
                "is_disruptive": false
            }
            
            5. Pronunciation Practice:
            Input: "shokusenki"
            Response: {
                "step": "evaluate_pronunciation",
                "response_text": "「しょ」の発音をもう一度練習しましょう。",
                "is_disruptive": false
            }
            
            6. Pronunciation Repeat Request:
            Input: "Can you say that again?"
            Response: {
                "step": "model_pronunciation",
                "response_text": "はい、もう一度言います：しょくせんき",
                "is_disruptive": false
            }
            
            7. Good Pronunciation:
            Input: "しょくせんき"
            Response: {
                "step": "next_word",
                "response_text": "素晴らしい発音です！次の言葉に進みましょう。",
                "is_disruptive": false
            }
            
            8. Off-topic but polite:
            Input: "How do you write this in kanji?"
            Response: {
                "step": "redirect",
                "response_text": "漢字は後で勉強しましょう。今は発音の練習を続けましょう。",
                "is_disruptive": false
            }
            
            9. Disruptive behavior:
            Input: "This is boring, I don't want to do this"
            Response: {
                "step": "redirect",
                "response_text": "レッスンを続けるには集中が必要です。一緒に頑張りましょう。",
                "is_disruptive": true
            }
            
            10. Third disruption:
            Input: (after 2 previous disruptions)
            Response: {
                "step": "end_lesson",
                "response_text": "申し訳ありませんが、レッスンを終了させていただきます。また集中できる時に勉強しましょう。",
                "is_disruptive": true
            }"""),
            HumanMessage(content=f"""
            Current state:
            - Word: {state['current_word']}
            - Current teaching step: {state['current_step']}
            - User input: {state['last_user_input']}
            - Previous disruptions: {len(state['disruption_history'])}
            """)
        ]
    )
    
    new_state = state.copy()
    
    if decision.is_disruptive:  # Now this field exists in the model
        new_state["disruption_history"].append({
            "step": state["current_step"],
            "input": state["last_user_input"],
            "timestamp": datetime.now().isoformat()
        })
        
    if len(new_state["disruption_history"]) >= 3:
        decision.step = "end_lesson"
        decision.response_text = "申し訳ありませんが、レッスンを終了させていただきます。"
    
       # Ensure decision.step is never empty
    if not decision.step:
        decision.step = "end"
    
    return {
        "decision": decision.step,
        "output": decision.response_text,
        **new_state
    }

# Teaching nodes
def give_context(state: State):
    """Introduce the lesson context"""
    result = llm.invoke(
        f"""Introduce the vocabulary lesson in Japanese.
        Current word: {state['current_word']}
        
        Example: 今日は便利な電化製品について勉強しましょう。
        """
    )
    return {"output": result.content}

def present_word(state: State):
    """Present the new word in a natural context/sentence"""
    result = llm.invoke(
        f"""Present the word {state['current_word']['word']} in a natural Japanese sentence.
        The sentence should clearly demonstrate the word's meaning.
        
        Example format:
        この文を聞いてください：
        食洗器で皿を洗います。
        
        Current word: {state['current_word']['word']}
        Reading: {state['current_word']['reading']}
        Meaning: {state['current_word']['meaning']}
        """
    )
    return {"output": result.content}

def give_hint(state: State):
    """Provide a contextual hint in Japanese"""
    result = llm.invoke(
        f"""Give a helpful hint in Japanese about the word's meaning.
        The hint should guide without directly giving the answer.
        
        Word: {state['current_word']['word']}
        Previous user guess: {state['last_user_input']}
        Meaning: {state['current_word']['meaning']}
        
        Example hints:
        - 台所で使う機械です。
        - 毎日の家事に役立つ電化製品です。
        
        Give a natural hint that builds on the user's partial understanding: {state['last_user_input']}
        """
    )
    return {
        "output": result.content,
        "meaning_attempts": state["meaning_attempts"] + 1
    }

def check_meaning(state: State):
    """Evaluate user's meaning comprehension"""
    result = llm.invoke(
        f"""Evaluate if the user understood the word's meaning.
        Word: {state['current_word']}
        User response: {state['last_user_input']}
        Attempts: {state['meaning_attempts']}
        
        Respond naturally in Japanese, encouraging further attempts if incorrect.
        """
    )
    return {"output": result.content}

def model_pronunciation(state: State):
    """Model the pronunciation for the user"""
    result = llm.invoke(
        f"""Model the pronunciation of {state['current_word']['word']}.
        Reading: {state['current_word']['reading']}
        
        Example: では、発音を練習しましょう。私の後について言ってください：しょくせんき
        """
    )
    return {"output": result.content}

# Add dedicated pronunciation evaluation node
def evaluate_pronunciation(state: State):
    """Evaluate user's pronunciation attempt"""
    result = llm.invoke(
        f"""Evaluate the user's pronunciation attempt.
        Target word: {state['current_word']['word']}
        Correct reading: {state['current_word']['reading']}
        User attempt: {state['last_user_input']}
        Current attempt number: {state['pronunciation_attempts']}
        
        Evaluate accuracy focusing on:
        - しょ sound accuracy
        - Overall rhythm
        - Pitch accent
        
        Example response for good attempt:
        "とても良い発音です！特に「しょ」の発音が自然です。"
        
        Example response for needs improvement:
        "惜しいですね。「しょ」の発音をもう一度練習しましょう。"
        """
    )
    return {
        "output": result.content,
        "pronunciation_attempts": state["pronunciation_attempts"] + 1
    }
