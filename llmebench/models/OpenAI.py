import os

import openai

import json

from itertools import chain

from llmebench.models.model_base import ModelBase



class OpenAIModelBase(ModelBase):
    """
    OpenAI Model interface. Can be used for models hosted on both OpenAI's platform and
    on Azure.

    Arguments
    ---------
    api_type : str
        Must be one of "openai" or "azure". If not provided, the implementation will try
        to induce it from environment variables `OPEN_API_TYPE`, `AZURE_*` or default to
        "openai"
    api_base : str
        URL where the model is hosted. Can be left as None for models hosted on OpenAI's
        platform. If not provided, the implementation will look at environment variables
        `OPENAI_API_BASE` or `AZURE_API_URL`
    api_version : str
        Version of the API to use. If not provided, the implementation will derive it
        from environment variables `OPENAI_API_VERSION` or `AZURE_API_VERSION`. Must be
        left as None for models hosted on OpenAI's platform
    api_key : str
        Authentication token for the API. If not provided, the implementation will derive it
        from environment variables `OPENAI_API_KEY` or `AZURE_API_KEY`.
    model_name : str
        Name of the model to use. If not provided, the implementation will derive it from
        environment variables `OPENAI_MODEL` or `AZURE_ENGINE_NAME`
    engine_name : str
        Alternative for `model_name`
    temperature : float
        Temperature value to use for the model. Defaults to zero for reproducibility.
    top_p : float
        Top P value to use for the model. Defaults to 0.95
    max_tokens : int
        Maximum number of tokens to pass to the model. Defaults to 800
    frequency_penalty : float
        Frequency Penalty to use for the model.
    presence_penalty : float
        Presence Penalty to use for the model.
    """

    def __init__(
        self,
        api_type=None,
        api_base=None,
        api_version=None,
        api_key=None,
        model_name=None,
        engine_name=None,
        temperature=0,
        top_p=0.95,
        max_tokens=800,
        frequency_penalty=0,
        presence_penalty=0,
        **kwargs
    ):
        # API parameters
        # Order of priority is:
        #   1. arguments to the constructor
        #   2. OPENAI_* env vars
        #   3. AZURE_* env vars
        azure_vars = self.read_azure_env_vars()
        openai_vars = self.read_openai_env_vars()

        api_type = (
            api_type or openai_vars["api_type"] or azure_vars["api_type"] or "openai"
        )
        api_base = api_base or openai_vars["api_base"] or azure_vars["api_base"]
        api_version = (
            api_version or openai_vars["api_version"] or azure_vars["api_version"]
        )
        api_key = api_key or openai_vars["api_key"] or azure_vars["api_key"]
        model_name = (
            model_name or engine_name or openai_vars["model"] or azure_vars["model"]
        )

        openai.api_type = api_type

        if api_type == "azure" and api_base is None:
            raise Exception(
                "API URL must be provided as model config or environment variable (`AZURE_API_BASE`)"
            )

        if api_base:
            openai.api_base = api_base

        if api_type == "azure" and api_version is None:
            raise Exception(
                "API version must be provided as model config or environment variable (`AZURE_API_VERSION`)"
            )

        if api_version:
            openai.api_version = api_version

        if api_key is None:
            raise Exception(
                "API Key must be provided as model config or environment variable (`OPENAI_API_KEY` or `AZURE_API_KEY`)"
            )

        openai.api_key = api_key

        self.model_params = {}

        if model_name is None:
            raise Exception(
                "Model/Engine must be provided as model config or environment variable `OPENAI_MODEL`/`AZURE_ENGINE_NAME`"
            )

        if api_type == "azure":
            self.model_params["engine"] = model_name
        else:
            self.model_params["model"] = model_name

        # GPT parameters
        self.model_params["temperature"] = temperature
        self.model_params["top_p"] = top_p
        self.model_params["max_tokens"] = max_tokens
        self.model_params["frequency_penalty"] = frequency_penalty
        self.model_params["presence_penalty"] = presence_penalty

        super(OpenAIModelBase, self).__init__(
            retry_exceptions=(openai.error.Timeout, openai.error.RateLimitError),
            **kwargs
        )

    @staticmethod
    def read_azure_env_vars():
        curr_api_type = None
        if "AZURE_ENGINE_NAME" in os.environ or "ENGINE_NAME" in os.environ:
            curr_api_type = "azure"
        return {
            "api_type": curr_api_type,
            "api_version": os.getenv("AZURE_API_VERSION"),
            "api_base": os.getenv("AZURE_API_URL"),
            "api_key": os.getenv("AZURE_API_KEY"),
            "model": os.getenv("AZURE_ENGINE_NAME", os.getenv("ENGINE_NAME")),
        }

    @staticmethod
    def read_openai_env_vars():
        return {
            "api_type": os.getenv("OPEN_API_TYPE"),
            "api_version": os.getenv("OPENAI_API_VERSION"),
            "api_base": os.getenv("OPENAI_API_BASE"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_MODEL"),
        }


class LegacyOpenAIModel(OpenAIModelBase):
    # defining a function to create the prompt from the system and user messages
    def create_prompt(self, system_message, messages):
        system_message_template = "<|im_start|>system\n{}\n<|im_end|>"
        message_template = "\n<|im_start|>{}\n{}\n<|im_end|>"

        prompt = system_message_template.format(system_message)

        for message in messages:
            prompt += message_template.format(message["sender"], message["text"])
        prompt += "\n<|im_start|>assistant\n"
        return prompt

    def summarize_response(self, response):
        """Returns the first reply, if available"""
        if (
            "choices" in response
            and isinstance(response["choices"], list)
            and len(response["choices"]) > 0
            and "text" in response["choices"][0]
        ):
            return response["choices"][0]["text"]

        return response

    def prompt(self, processed_input):
        """
        OpenAI API Completion implementation

        .. warning::
        This implementation is deprecated and will be removed in future versions. Use
        `OpenAIModel` instead.

        Arguments
        ---------
        processed_input : dict
            Must be a dictionary with two keys; "system_message" with a string
            value, and "messages" with a list value, where each element is a
            dictionary with two string-valued keys, "sender" and "text".

        Returns
        -------
        response : OpenAI API response
            Response from the openai python library
        """
        system_message = processed_input["system_message"]
        messages = processed_input["messages"]
        prompt = self.create_prompt(system_message, messages)
        response = openai.Completion.create(
            prompt=prompt, stop=["<|im_end|>"], **self.model_params
        )

        return response



class OpenAIModel(OpenAIModelBase):
    def summarize_response(self, response):
        """Returns the first reply from the "assistant", if available"""
        if (
            "choices" in response
            and isinstance(response["choices"], list)
            and len(response["choices"]) > 0
            and "message" in response["choices"][0]
            and "content" in response["choices"][0]["message"]
            and response["choices"][0]["message"]["role"] == "assistant"
        ):
            return response["choices"][0]["message"]["content"]

        return response

    def last_prompt_fever(self, data_row):
        question = data_row["question"]
        try:
            contexts = data_row["context"]["sentences"]
            paragraphs = [''.join(docs) for docs in contexts]
        except:
            try:
                paragraphs = [ ctx["text"] for ctx in  data_row['ctxs'] ]
            except:
                paragraphs = data_row['text']

                
        prompt_string = (
            f"Claim: {question}\nContext: {paragraphs}"
            f"Output josn:\n\n"
        )

        system_string = (
            f"You are a text classification agent. Given a context and a claim, please give a judgement to the claim ('SUPPORTS' or 'REFUTES') based on the context. " 
            # f"Generate the answer in a json output format with 'answer' tag, an 'evidence_from_context' tag and a 'step_by_step_reasoning' tag. "
            f"Generate the answer in a json output format with 'answer' tag and an 'evidence_and_explanation' tag "
            f"Your answer must be the either ('SUPPORTS' or 'REFUTES') depeneding on whether the claim is supported or refuted by the context. "
            )
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]
    

    def last_prompt_tqa(self, data_row):
        question = data_row["question"]
        try:
            contexts = data_row["context"]["sentences"]
            paragraphs = [''.join(docs) for docs in contexts]
        except:
            try:
                paragraphs = [ ctx["text"] for ctx in  data_row['ctxs'] ]
            except:
                paragraphs = data_row['text']

                
        prompt_string = (
            f"Question: {question}\nContext: {paragraphs}"
            f"Output josn:\n\n"
        )

        system_string = (
            f"You are a question answering agent. Given a context and a question, your task is to answer the question based on the context. " 
            f"Generate the answer in a json output format with 'answer' tag and an 'evidence_and_explanation' tag "
            f"Instead of a full sentence, your answer must be the shortest word or phrase or named enitity. "
            f"Some example outputs 'answer' are: yes; no; Ibn Sina; Doha, Qatar; 2,132 seats, Los Angeles, California etc.,. Please make sure it's valid json. " 
          )
    
        
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]
    
    def last_prompt_eli5(self, data_row):
        question = data_row["question"]
        try:
            contexts = data_row["context"]["sentences"]
            paragraphs = [''.join(docs) for docs in contexts]
        except:
            try:
                paragraphs = [ ctx["text"] for ctx in  data_row['ctxs'] ]
            except:
                paragraphs = data_row['text']

                
        prompt_string = (
            f"Question: {question}\nContext: {paragraphs}"
            f"Output josn:\n\n"
        )

        system_string = (
            f"You are a question answering agent. Given a context and a question, your task is to long answer the question based on the context. " 
            f"Generate the answer in a json output format with 'answer' tag and an 'evidence_and_explanation' tag "
            f"Answer this as for a 5 years old child. So your answer must be very very long and very very detailed.  "
            ) 
    
        
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]
    
    def last_prompt_hotpotqa_huggingface(self, data_row):
        question = data_row["question"]
        
        contexts = data_row["context"]["sentences"]
        paragraphs = [''.join(docs) for docs in contexts]
        

                
        prompt_string = (
            f"Question: {question}\nContext: {paragraphs}"
            f"Output josn:\n\n"
        )

        system_string = (
            f"You are a question answering agent. Given a context and a question, your task is to answer the question based on the context. " 
            f"Generate the answer in a json output format with 'answer' tag and an 'evidence_and_explanation' tag "
            f"Instead of a full sentence, your answer must be the shortest word or phrase or named enitity. "
            f"Some example outputs 'answer' are: yes; no; Ibn Sina; Doha, Qatar; 2,132 seats, Los Angeles, California etc.,. Please make sure it's valid json. " 
          )
        
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]
    
    
    def last_prompt_logiqa_huggingface(self, data_row):
        question = data_row["query"]
        
        paragraphs = data_row["context"]
        
        options =  data_row["options"]

        import pdb
        # pdb.set_trace()
        prompt_string = (
            f"Question: {question}\nContext: {paragraphs}\nOptions: {options}"
            f"Output josn:\n\n"
        )

        system_string = (
            f"You are a contextualized logical question answering agent. Given a context, a question, and four options your task is to select the index of the option [0,1,2,3] that best answers the the question based on the context. " 
            f"Generate the answer in a json output format with 'answer' tag and an 'evidence_and_explanation' tag "
           )
        
        
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]
    
    def last_prompt_drop_huggingface(self, data_row):
        question = data_row["question"]
        
        paragraphs = data_row["passage"]

        
        prompt_string = (
        f"Question: {question}\nContext: {paragraphs}"
        f"Output josn:\n\n"
        )
        

        system_string = (
            f"You are a question answering agent. Given a context and a question, your task is to answer the question based on the context. " 
            f"Generate the answer in a json output format with 'answer' tag and an 'evidence_and_explanation' tag "
           )
        
        
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]
    

    def cot_prompt_logiqa_huggingface(self, data_row):
        question = data_row["query"]
        
        paragraphs = data_row["context"]
        
        options =  data_row["options"]

        import pdb
        # pdb.set_trace()
        prompt_string = (
            f"Question: {question}\nContext: {paragraphs}\nOptions: {options}"
            f"Output josn:\n\n"
        )

        system_string = (
            f"You are a contextualized logical question answering agent. Given a context, a question, and four options your task is to select the index of the option [0,1,2,3] that best answers the the question based on the context. " 
            f"Think step by step and generate the answer in a json output format with 'answer' tag and 'step_by_step_reasoning' tag "
           )
        
        
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]

    def cot_prompt_drop_huggingface(self, data_row):
        question = data_row["question"]
        
        paragraphs = data_row["passage"]

        
        prompt_string = (
        f"Question: {question}\nContext: {paragraphs}"
        f"Output josn:\n\n"
        )
        

        prompt_string = (
        f"Question: {question}\nContext: {paragraphs}"
        f"Output josn:\n\n"
        )
        

        system_string = (
            f"You are a question answering agent. Given a context and a question, your task is to answer the question based on the context. " 
            f"Think step by step and generate the answer in a json output format with 'answer' tag and 'step_by_step_reasoning' tag "
           )
        
        
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]

    def last_prompt_nq(self, data_row):
        question = data_row["question"]
        try:
            contexts = data_row["context"]["sentences"]
            paragraphs = [''.join(docs) for docs in contexts]
        except:
            try:
                paragraphs = [ ctx["text"] for ctx in  data_row['ctxs'] ]
            except:
                paragraphs = data_row['text']

                
        prompt_string = (
            f"Question: {question}\nContext: {paragraphs}"
            f"Output josn:\n\n"
        )

        system_string = (
            f"You are a question answering agent. Given a context and a question, your task is to answer the question based on the context. " 
            f"Generate the answer in a json output format with 'answer' tag and an 'evidence_and_explanation' tag "
            f"Instead of a full sentence, your answer must be the shortest word or phrase or named enitity. "
            f"Some example outputs 'answer' are: yes; no; Ibn Sina; Doha, Qatar; 2,132 seats, Los Angeles, California etc.,. Please make sure it's valid json. " 
          )
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]
    
    def last_prompt_wow(self, data_row):
        question = data_row["question"]
        try:
            contexts = data_row["context"]["sentences"]
            paragraphs = [''.join(docs) for docs in contexts]
        except:
            try:
                paragraphs = [ ctx["text"] for ctx in  data_row['ctxs'] ]
            except:
                paragraphs = data_row['text']

                
        prompt_string = (
            f"Question: {question}\nContext: {paragraphs}"
            f"Output josn:\n\n"
        )

        system_string = (
            f"You are a helpful AI assistant for Knowledge-Grounded Dialog Generation.  Given a context and a query, please give the next-turn response to the query based on the context. " 
            f"Generate your next-turn response in a json output format with 'answer' tag and an 'evidence_and_explanation' tag "
            )
        
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]
    
    def direct_prompt_fever(self, data_row):
    
       
        question = data_row["question"]
        try:
            contexts = data_row["context"]["sentences"]
            paragraphs = [''.join(docs) for docs in contexts]
        except:
            paragraphs = [ ctx["text"] for ctx in  data_row['ctxs'] ]
        
        prompt_string = (
            f"Claim: {question}\nContext: {paragraphs}"
            f"Output josn:\n\n"
        )

        system_string = (
        f"You are a text classification agent. Given a context and a claim, please give a judgement to the claim ('SUPPORTS' or 'REFUTES') based on the context. " 
        f"Generate the answer in a json output format with 'answer' tag "
        f"Your answer must be the either ('SUPPORTS' or 'REFUTES') depending on whether the claim is supported or refuted by the context. "
        )
        
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]
    def direct_prompt(self, data_row):
        question = data_row["question"]
        try:
            contexts = data_row["context"]["sentences"]
            paragraphs = [''.join(docs) for docs in contexts]
        except:
            paragraphs = [ ctx["text"] for ctx in  data_row['ctxs'] ]
        
        prompt_string = (
            f"Question: {question}\nContext: {paragraphs}"
            f"Output josn:\n\n"
        )

        system_string = (
            f"You are a question answering agent. Given a context and a question, your task is to answer the question based on the context. " 
            f"Generate the answer in a json output format with 'answer' tag "
            f"Instead of a full sentence, your answer must be the shortest word or phrase or named enitity. "
            f"Some example outputs 'answer' are: yes; no; Ibn Sina; Doha, Qatar; 2,132 seats, Los Angeles, California etc.,. Please make sure it's valid json. " 
          )
        
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]
    
    def cot_prompt_fever(self, data_row):
    
       
        question = data_row["question"]
      
        try:
            contexts = data_row["context"]["sentences"]
            paragraphs = [''.join(docs) for docs in contexts]
        except:
            paragraphs = [ ctx["text"] for ctx in  data_row['ctxs'] ]
        
        prompt_string = (
            f"claim: {question}\context: {paragraphs}"
            f"Output josn:\n\n"
        )

        
        system_string = (
            f"You are a text classification agent. Given a context and a claim, please give a judgement to the claim ('SUPPORTS' or 'REFUTES') based on the context. " 
            f"Think step by step and generate the answer in a json output format with 'answer' tag and 'step_by_step_reasoning' tag "
            f"Your answer must be the either ('SUPPORTS' or 'REFUTES') depending on whether the claim is supported or refuted by the context. "
            )
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]
    def cot_prompt_wow(self, data_row):
    
       
        question = data_row["question"]
      
        try:
            contexts = data_row["context"]["sentences"]
            paragraphs = [''.join(docs) for docs in contexts]
        except:
            paragraphs = [ ctx["text"] for ctx in  data_row['ctxs'] ]
        
        prompt_string = (
            f"claim: {question}\context: {paragraphs}"
            f"Output josn:\n\n"
        )

        
        system_string = (
            f"You are a helpful AI assistant. Given a context and a query, please give the next-turn response to the query based on the context.  " 
            f"Think step by step and generate the response in a json output format with 'answer' tag and 'step_by_step_reasoning' tag "
            )
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]



    def cot_prompt(self, data_row):
    
       
        question = data_row["question"]
      
        try:
            contexts = data_row["context"]["sentences"]
            paragraphs = [''.join(docs) for docs in contexts]
        except:
            paragraphs = [ ctx["text"] for ctx in  data_row['ctxs'] ]
        
        prompt_string = (
             f"Question: {question}\nContext: {paragraphs}"
            f"Output josn:\n\n"
        )

        system_string = (
            f"You are a question answering agent. Given a context and a question, your task is to answer the question based on the context. " 
            f"Think step by step and generate the answer in a json output format with 'answer' tag and 'step_by_step_reasoning' tag "
            f"Instead of a full sentence, your answer must be the shortest word or phrase or named enitity. "
            f"Some example outputs 'answer' are: yes; no; Ibn Sina; Doha, Qatar; 2,132 seats, Los Angeles, California etc.,. Please make sure it's valid json. " 
          )
        
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]
    
    def cot_prompt_eli5(self, data_row):
    
       
        question = data_row["question"]
      
        try:
            contexts = data_row["context"]["sentences"]
            paragraphs = [''.join(docs) for docs in contexts]
        except:
            paragraphs = [ ctx["text"] for ctx in  data_row['ctxs'] ]
        
        prompt_string = (
             f"Question: {question}\nContext: {paragraphs}"
            f"Output josn:\n\n"
        )

        system_string = (
            f"You are a question answering agent. Given a context and a question, your task is to answer the question based on the context. " 
            f"Think step by step and generate the answer in a json output format with 'answer' tag and 'step_by_step_reasoning' tag "
            f"Answer me for a 5 years old child. So your answer must be long and very detailed.  "
            )
        
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]
    
    def evidence_prompt(self, data_row):
    
        question = data_row["question"]
        try:
            contexts = data_row["context"]["sentences"]
            paragraphs = [''.join(docs) for docs in contexts]
        except:
            paragraphs = [ ctx["text"] for ctx in  data_row['ctxs'] ]


        paragraphs = [''.join(docs) for docs in contexts]
        
        prompt_string = (
            f"Question: {question}\nContext: {paragraphs}"
            f"Output josn:\n\n"
        )

        system_string = (
            f"You are a question answering agent. "
            f"Given a context and a question, your task is provide me with the relevant sentences from the context to answer the question" 
            f"Your output must be in a json output format with an 'evidence_sentences' tag "
        )
        return [
            {
                "role": "system",
                "content": system_string,
            },
            {"role": "user", "content": prompt_string},
        ]


    def get_sentences_from_json_response(self, ctx):
        sentences = []
        """
        Response is mainly in one of these format
        # case-1: {'sentences': [{'raw_sentence': 'Tyler Bates (born June 5, 1965) is an American musician, music producer, and composer for films, television, and video games.', 'resolved_sentence': 'Tyler Bates (born June 5, 1965) is an American musician, music producer, and composer for films, television, and video games.', 'changes': [], 'coref_after_article_the': [], 'any_parphrasing_or_other_changes_done': [], 'valid_json_output': True}, {'raw_sentence': 'Much of his work is in the action and horror film genres, with films like "Dawn of the Dead, 300, Sucker Punch," and "John Wick."', 'resolved_sentence': 'Much of Tyler Bates\'s work is in the action and horror film genres, with films like "Dawn of the Dead, 300, Sucker Punch," and "John Wick."', 'changes': [{'old': 'his', 'new': "Tyler Bates's", 'span': [10, 20]}], 'coref_after_article_the': [], 'any_parphrasing_or_other_changes_done': [], 'valid_json_output': True}, {'raw_sentence': 'He has collaborated with directors like Zack Snyder, Rob Zombie, Neil Marshall, William Friedkin, Scott Derrickson, and James Gunn.', 'resolved_sentence': 'Tyler Bates has collaborated with directors like Zack Snyder, Rob Zombie, Neil Marshall, William Friedkin, Scott Derrickson, and James Gunn.', 'changes': [{'old': 'He', 'new': 'Tyler Bates', 'span': [0, 2]}], 'coref_after_article_the': [], 'any_parphrasing_or_other_changes_done': [], 'valid_json_output': True}, {'raw_sentence': 'With Gunn, he has scored every one of the director\'s films; including "Guardians of the Galaxy", which became one of the highest grossing domestic movies of 2014, and its 2017 sequel.', 'resolved_sentence': 'With James Gunn, Tyler Bates has scored every one of the director\'s films; including "Guardians of the Galaxy", which became one of the highest grossing domestic movies of 2014, and its 2017 sequel.', 'changes': [{'old': 'he', 'new': 'Tyler Bates', 'span': [5, 7]}], 'coref_after_article_the': [], 'any_parphrasing_or_other_changes_done': [], 'valid_json_output': True}, {'raw_sentence': 'In addition, he is also the lead guitarist of the American rock band Marilyn Manson, and produced its albums "The Pale Emperor" and "Heaven Upside Down".', 'resolved_sentence': 'In addition, Tyler Bates is also the lead guitarist of the American rock band Marilyn Manson, and produced Marilyn Manson\'s albums "The Pale Emperor" and "Heaven Upside Down".', 'changes': [{'old': 'he', 'new': 'Tyler Bates', 'span': [0, 2]}, {'old': 'its', 'new': "Marilyn Manson's", 'span': [54, 57]}], 'coref_after_article_the': [], 'any_parphrasing_or_other_changes_done': [], 'valid_json_output': True}]}
        # case-2: {'sentence1': {'raw_sentence': 'Doctor Strange is a 2016 American superhero film based on the Marvel Comics character of the same name, produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures.', 'resolved_sentence': 'The 2016 American superhero film Doctor Strange is based on the Marvel Comics character of the same name, and it was produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures.', 'changes': [{'old': 'Doctor Strange', 'new': 'The 2016 American superhero film Doctor Strange'}], 'coref_after_article_the': '', 'any_parphrasing_or_other_changes_done': 'Changed the sentence structure to make it grammatically correct.', 'valid_json_output': True}, 'sentence2': {'raw_sentence': 'It is the fourteenth film of the Marvel Cinematic Universe (MCU).', 'resolved_sentence': 'The 2016 American superhero film Doctor Strange is the fourteenth film of the Marvel Cinematic Universe (MCU).', 'changes': [{'old': 'It', 'new': 'The 2016 American superhero film Doctor Strange'}], 'coref_after_article_the': '', 'any_parphrasing_or_other_changes_done': '', 'valid_json_output': True}, 'sentence3': {'raw_sentence': 'The film was directed by Scott Derrickson, who wrote it with Jon Spaihts and C. Robert Cargill, and stars Benedict Cumberbatch as Stephen Strange, along with Chiwetel Ejiofor, Rachel McAdams, Benedict Wong, Michael Stuhlbarg, Benjamin Bratt, Scott Adkins, Mads Mikkelsen, and Tilda Swinton.', 'resolved_sentence': 'Scott Derrickson directed the 2016 American superhero film Doctor Strange, and he wrote it with Jon Spaihts and C. Robert Cargill. The film stars Benedict Cumberbatch as Stephen Strange, along with Chiwetel Ejiofor, Rachel McAdams, Benedict Wong, Michael Stuhlbarg, Benjamin Bratt, Scott Adkins, Mads Mikkelsen, and Tilda Swinton.', 'changes': [{'old': 'The film', 'new': 'the 2016 American superhero film Doctor Strange'}], 'coref_after_article_the': '', 'any_parphrasing_or_other_changes_done': 'Changed the sentence structure to make it grammatically correct.', 'valid_json_output': True}, 'sentence4': {'raw_sentence': 'In "Doctor Strange", surgeon Strange learns the mystic arts after a career-ending car accident.', 'resolved_sentence': 'In the 2016 American superhero film Doctor Strange, Stephen Strange learns the mystic arts after a career-ending car accident.', 'changes': [{'old': 'surgeon Strange', 'new': 'Stephen Strange'}], 'coref_after_article_the': '', 'any_parphrasing_or_other_changes_done': '', 'valid_json_output': True}}
        # case-3: [{'raw_sentence': 'Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood.', 'resolved_sentence': 'The 1994 American biographical period comedy-drama film Ed Wood was directed and produced by Tim Burton, and Johnny Depp starred as cult filmmaker Ed Wood.', 'changes': [{'old': 'Ed Wood', 'new': 'The 1994 American biographical period comedy-drama film Ed Wood'}, {'old': 'Johnny Depp', 'new': 'Johnny Depp'}, {'old': 'Ed Wood', 'new': 'cult filmmaker Ed Wood'}], 'coref_after_article_the': [{'old': 'the supporting cast', 'new': 'the supporting cast supporting cast'}], 'any_parphrasing_or_other_changes_done': 'None', 'valid_json_output': True}, {'raw_sentence': "The film concerns the period in Wood's life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau.", 'resolved_sentence': "The 1994 American biographical period comedy-drama film Ed Wood concerns the period in Ed Wood's life when Ed Wood made his best-known films as well as Ed Wood's relationship with actor Bela Lugosi, played by Martin Landau.", 'changes': [{'old': "Wood's", 'new': "Ed Wood's"}, {'old': 'he', 'new': 'Ed Wood'}, {'old': 'his', 'new': "Ed Wood's"}, {'old': 'Ed Wood', 'new': 'Ed Wood'}, {'old': 'his', 'new': "Ed Wood's"}, {'old': 'Bela Lugosi', 'new': 'actor Bela Lugosi'}, {'old': 'Martin Landau', 'new': 'Martin Landau'}], 'coref_after_article_the': [], 'any_parphrasing_or_other_changes_done': 'None', 'valid_json_output': True}, {'raw_sentence': 'Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast.', 'resolved_sentence': 'Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast of the 1994 American biographical period comedy-drama film Ed Wood.', 'changes': [{'old': 'the supporting cast', 'new': 'the supporting cast of the 1994 American biographical period comedy-drama film Ed Wood'}], 'coref_after_article_the': [], 'any_parphrasing_or_other_changes_done': 'None', 'valid_json_output': True}]
        # case-4: {'raw_sentence': 'Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.', 'resolved_sentence': 'Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.', 'changes': [], 'coref_after_article_the': [], 'any_parphrasing_or_other_changes_done': [], 'valid_json_output': True}
        """
        if isinstance(ctx, list):
            for ct in ctx:
                sentences.append(ct["resolved_sentence"]) #case 3
        else:
            if "resolved_sentence" in ctx:
                sentences.append(ctx["resolved_sentence"]) #case 4
            else:
                if len(ctx)>1:
                    for _, js in ctx.items():
                        sentences.append(js["resolved_sentence"]) #case 2
                else:
                    for _, js in ctx.items():
                        for j in js:
                            sentences.append(j["resolved_sentence"]) #case 1
        return sentences

    
    def coref_prompt(self, contexts):
        paragraphs = [''.join(docs) for docs in contexts]
        new_contexts = []

        for i, pr in enumerate(paragraphs):
            # pdb.set_trace()
            prompt_string = (
                f"paragraph: {pr}\n\n"
            )

            system_string = (
                f"You are an AI assistant for corefrence resolution."
                f"You will be given a paragraph and you will produce a new paragraph replacing the pronouns with the first/original reference within the paragraph "
                f"output is the sentences in the paragraphs in a (must) valid json format each with mendatory six following tags.  "
                f"1. raw_sentence: original sentence in the given paraphraph; "
                f"2. resolved_sentence: new sentence after the coreference resolution. "
                f"3. changes: replacement old and new span in the sentence.  "
                f"4. coref_after_article_the: any coreference that came after article 'the' e.g., 'the film' should be replaced with 'the film film_name' " 
                f"5. any_parphrasing_or_other_changes_done: any changes made or parapharsing done other than coreference replacemnets. "
                f"6. valid_json_output: if the output can be loaded with json.loads() "
            )
        
            msg = [
                { "role": "system", "content": system_string,},
                {"role": "user", "content": prompt_string},
            ]
            response = openai.ChatCompletion.create(
                    messages=msg, **self.model_params
            )
            # pdb.set_trace()
            try:
                ctx = json.loads(response["choices"][0]["message"]["content"])
            except:
                msg = [
                    { "role": "system", "content": "You are json formatter who can fix json format erors. Given an broken json, you will fix all commas and quotes and alll errors and provide oner single valid json object. ",},
                    {"role": "user", "content": response["choices"][0]["message"]["content"]+"\n output json: " },
                ]
                response = openai.ChatCompletion.create(
                    messages=msg, **self.model_params
                )
                try:
                    ctx = json.loads(response["choices"][0]["message"]["content"])
                except:
                    new_contexts.append([pr])
                    continue
                
            new_contexts.append(self.get_sentences_from_json_response(ctx))
            
        return new_contexts

    def rank_sentences(self, question, contexts):
        sentences = list(chain(*contexts))
        sent_relevance = {}
        for sent in  sentences:
            prompt_string = (
                f"Question: {question}\n context: {sent}"
                f"Output josn:\n\n"
            )

            system_string = (
                f"You are a context retrieval/ranking model for a single or multi-hop question answering that means the context can provide full or partial information relevant for answering the question.  "
                f"Your input will be given a question and a context as input and you will need to compute the 'retrieval_score' for the corresponding context with respect to the given question. "
                f"your retrieval_score should be 3 demical point precision and the output should be a json with two tags:  'retrieval_score' and 'explantaion' "

                ''' f"You are a context retrieval/ranking model for a single or multi-hop question answering. "
                f"For each input context, you need to compute its retrieval score with respect to the given question. "
                f"Your output must be a json of tuples. ('context_index', 'retrieval_score') for each input context. "
                f"you can sort tuples in order to their score from high to low. retrieval_score should be 2 demical point precision. '''

            )
            msg =  [
                { "role": "system", "content": system_string},
                {"role": "user", "content": prompt_string},
            ]
            
            response = openai.ChatCompletion.create( messages=msg, **self.model_params )   
            # pdb.set_trace()
            
            score = json.loads(response["choices"][0]["message"]["content"])['retrieval_score']
            sent_relevance.update({sent: score})
            
            
        
        return sent_relevance


    def prompt_main_three_agent_coref_rank_qa(self, processed_input):
        coref_contexts = self.coref_prompt(processed_input["context"]["sentences"])
        assert len(coref_contexts) == len(processed_input["context"]["sentences"])

        sentence_score = self.rank_sentences(processed_input["question"], coref_contexts)
        ranked_sentences = sorted(sentence_score.items(), key=lambda item: float(item[1]))[::-1]
        ranked_sentences =  [ x[0] for x in ranked_sentences]
        
        processed_input["context"]["sentences"] = ranked_sentences
     
        last_prompt_msg = self.cot_prompt(processed_input)
        response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        return response
    
    def prompt_main_four_agent_coref_rank_evidence_qa_with_ori(self, processed_input):
        coref_contexts = self.coref_prompt(processed_input["context"]["sentences"])
        assert len(coref_contexts) == len(processed_input["context"]["sentences"])

        sentence_score = self.rank_sentences(processed_input["question"], coref_contexts)
        ranked_sentences = sorted(sentence_score.items(), key=lambda item: float(item[1]))[::-1]
        ranked_sentences =  [ x[0] for x in ranked_sentences]

        ori_sentences = processed_input["context"]["sentences"]
        processed_input["context"]["sentences"] = ranked_sentences

        last_prompt_msg = self.evidence_prompt(processed_input)
        evidence = json.loads(response["choices"][0]["message"]["content"])['evidence_context']
        new_ranked_sentences = [evidence] + ori_sentences
        processed_input["context"]["sentences"] = new_ranked_sentences
        last_prompt_msg = self.cot_prompt(processed_input)
        response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        return response
    
    def prompt_main_four_agent_coref_rank_evidence_qa_with_reranked(self, processed_input):
        coref_contexts = self.coref_prompt(processed_input["context"]["sentences"])
        assert len(coref_contexts) == len(processed_input["context"]["sentences"])

        sentence_score = self.rank_sentences(processed_input["question"], coref_contexts)
        ranked_sentences = sorted(sentence_score.items(), key=lambda item: float(item[1]))[::-1]
        ranked_sentences =  [ x[0] for x in ranked_sentences]

        ori_sentences = processed_input["context"]["sentences"]
        processed_input["context"]["sentences"] = ranked_sentences

        last_prompt_msg = self.evidence_prompt(processed_input)
        evidence = json.loads(response["choices"][0]["message"]["content"])['evidence_context']
        new_ranked_sentences = [evidence] + ranked_sentences
        processed_input["context"]["sentences"] = new_ranked_sentences
        last_prompt_msg = self.cot_prompt(processed_input)
        response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        return response
    
    def prompt_main_four_agent_coref_rank_evidence_qa(self, processed_input):
        coref_contexts = self.coref_prompt(processed_input["context"]["sentences"])
        assert len(coref_contexts) == len(processed_input["context"]["sentences"])

        sentence_score = self.rank_sentences(processed_input["question"], coref_contexts)
        ranked_sentences = sorted(sentence_score.items(), key=lambda item: float(item[1]))[::-1]
        ranked_sentences =  [ x[0] for x in ranked_sentences]

        processed_input["context"]["sentences"] = ranked_sentences

        last_prompt_msg = self.evidence_prompt(processed_input)
        evidence = json.loads(response["choices"][0]["message"]["content"])['evidence_context']
        new_ranked_sentences = [evidence]
        processed_input["context"]["sentences"] = new_ranked_sentences
        last_prompt_msg = self.cot_prompt(processed_input)
        response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        return response
    
    def prompt_main_single_agent_evidenPrompt2gen(self, processed_input):
        evidence_prompt_msg = self.evidence_prompt(processed_input)
        response = openai.ChatCompletion.create(messages=evidence_prompt_msg, **self.model_params)
        
        print(json.loads(response["choices"][0]["message"]["content"]))
        
        evidence = json.loads(response["choices"][0]["message"]["content"])['evidence_sentences']
        new_ranked_sentences = [evidence]
        processed_input["context"]["sentences"] = new_ranked_sentences
        last_prompt_msg = self.last_prompt(processed_input)
        response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        return response
    
    def prompt_main_single_agent_eviden_only(self, processed_input):
        last_prompt_msg = self.last_prompt(processed_input)
        response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        return response
    
    def prompt_main_single_agent_direct(self, processed_input):
        #direct
        last_prompt_msg = self.direct_prompt(processed_input)
        response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        return response

    def prompt_main_single_agent_cot(self, processed_input):
        #cot
        try:
            last_prompt_msg = self.cot_prompt(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        except:
            # print("Trying direct as none of e2g or cot works")
            last_prompt_msg = self.direct_prompt(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        return response
    
    def prompt_main_single_agent_cot_hotpotqa_huggingface(self, processed_input):
        #cot
        try:
            last_prompt_msg = self.cot_prompt(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        except:
            # print("Trying direct as none of e2g or cot works")
            last_prompt_msg = self.direct_prompt(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        return response
    
    def prompt_main_single_agent_cot_hotpotqa_huggingface_llama2(self, processed_input):
        # import pdb
        # pdb.set_trace()
        try:
            last_prompt_msg = self.cot_prompt(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        except:
            # print("Trying direct as none of e2g or cot works")
            last_prompt_msg = self.direct_prompt(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        return response
    
    def prompt_main_single_agent_cot_tqa(self, processed_input):
        #cot
        try:
            last_prompt_msg = self.cot_prompt(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        except:
            # print("Trying direct as none of e2g or cot works")
            last_prompt_msg = self.direct_prompt(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        return response
    
    def prompt_main_single_agent_cot_nq(self, processed_input):
        #cot
        try:
            last_prompt_msg = self.cot_prompt(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        except:
            # print("Trying direct as none of e2g or cot works")
            last_prompt_msg = self.direct_prompt(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        return response
    
    def prompt_main_single_agent_cot_eli5(self, processed_input):
        #cot
        last_prompt_msg = self.cot_prompt_eli5(processed_input)
        response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        
        return response
    
    def prompt_main_single_agent_cot_fever(self, processed_input):
        #cot
        try:
            last_prompt_msg = self.cot_prompt_fever(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        except:
            # print("Trying direct as none of e2g or cot works")
            last_prompt_msg = self.direct_prompt_fever(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        return response
    def prompt_main_single_agent_cot_wow(self, processed_input):
        #cot
        last_prompt_msg = self.cot_prompt_wow(processed_input)
        response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        
        return response
    
    def prompt_main_single_agent_cot_logiqa_huggingface(self, processed_input):
        #cot
        last_prompt_msg = self.cot_prompt_logiqa_huggingface(processed_input)
        response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        
        return response
    def prompt_main_single_agent_cot_drop_huggingface(self, processed_input):
        #cot
        import pdb
        # pdb.set_trace()
        last_prompt_msg = self.cot_prompt_drop_huggingface(processed_input)
        response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        
        return response
    
    def prompt_main_single_agent_two_cot(self, processed_input):
        evidence_prompt_msg = self.cot_prompt(processed_input)
        response = openai.ChatCompletion.create(messages=evidence_prompt_msg, **self.model_params)
        evidence = json.loads(response["choices"][0]["message"]["content"])['step_by_step_reasoning']
        try:
            processed_input["context"]["sentences"] = [evidence]
        except:
            processed_input["ctxs"]=[{"text":evidence}]
        last_prompt_msg = self.cot_prompt(processed_input)
        response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        return response

    def prompt_main_single_agent_estep_only(self, processed_input):
        last_prompt_msg = self.last_prompt(processed_input)
        response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        return response
    
    def prompt_main_single_agent_e2g_fever(self, processed_input):
        #ours
        try: 
            # if any crashes occurs
            evidence_prompt_msg = self.last_prompt_fever(processed_input)
            response = openai.ChatCompletion.create(messages=evidence_prompt_msg, **self.model_params)
            content = json.loads(response["choices"][0]["message"]["content"])
            evidence = content['evidence_and_explanation']
            try:
                processed_input["context"]["sentences"] = [evidence]
            except:
                processed_input["ctxs"]=[{"text":evidence}]
                last_prompt_msg = self.last_prompt_fever(processed_input)
                response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        except:
            try:
                return self.prompt_main_single_agent_cot_fever(processed_input)
            except: 
                return self.prompt_main_single_agent_direct_fever(processed_input)
        return response
    
    def prompt_main_single_agent_e2g_tqa(self, processed_input):
        #ours
        try: 
            # if any crashes occurs
            evidence_prompt_msg = self.last_prompt_tqa(processed_input)
            response = openai.ChatCompletion.create(messages=evidence_prompt_msg, **self.model_params)
            content = json.loads(response["choices"][0]["message"]["content"])
            evidence = content['evidence_and_explanation']
            processed_input["ctxs"]=[{"text":evidence}] 
            last_prompt_msg = self.last_prompt_tqa(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        except:
            try:
                return self.prompt_main_single_agent_cot(processed_input)
            except: 
                return self.prompt_main_single_agent_direct(processed_input)
        return response
    
    def prompt_main_single_agent_e2g_eli5(self, processed_input):
        #ours
        try: 
            # if any crashes occurs
            evidence_prompt_msg = self.last_prompt_eli5(processed_input)
            response = openai.ChatCompletion.create(messages=evidence_prompt_msg, **self.model_params)
            content = json.loads(response["choices"][0]["message"]["content"])
            evidence = content['answer'] 
            processed_input["ctxs"]=[{"text":evidence}] 
            last_prompt_msg = self.last_prompt_eli5(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        except:
            return self.prompt_main_single_agent_cot_eli5(processed_input)
            
        return response
    
    def prompt_main_single_agent_e2g_hotpotqa_huggingface(self, processed_input):
        #ours
        try: 
            # if any crashes occurs
            evidence_prompt_msg = self.last_prompt_hotpotqa_huggingface(processed_input)
            response = openai.ChatCompletion.create(messages=evidence_prompt_msg, **self.model_params)
            content = json.loads(response["choices"][0]["message"]["content"])
            evidence = content['evidence_and_explanation'] 
            # original_context = processed_input["context"]["sentences"]
            processed_input["context"]["sentences"] = [evidence] 
            last_prompt_msg = self.last_prompt_hotpotqa_huggingface(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        except:
            try:
                return self.prompt_main_single_agent_cot(processed_input)
            except: 
                return self.prompt_main_single_agent_direct(processed_input)
        return response
    
    def prompt_main_single_agent_e2g_logiqa_huggingface(self, processed_input):
        #ours
      
        # if any crashes occurs
        evidence_prompt_msg = self.last_prompt_logiqa_huggingface(processed_input)
        response = openai.ChatCompletion.create(messages=evidence_prompt_msg, **self.model_params)
        content = json.loads(response["choices"][0]["message"]["content"])
        evidence = content['evidence_and_explanation'] 
        # original_context = processed_input["context"]["sentences"]
        processed_input["context"] = evidence #+"\n"+processed_input["context"]
        last_prompt_msg = self.last_prompt_logiqa_huggingface(processed_input)
        response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        
        return response

    def prompt_main_single_agent_e2g_drop_huggingface(self, processed_input):
        #ours
      
        # if any crashes occurs
        # if "number" in processed_input["type"]:
        #     last_prompt_msg = self.cot_prompt_drop_huggingface(processed_input)
        #     response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        #     return response
        
        evidence_prompt_msg = self.last_prompt_drop_huggingface(processed_input)
        response = openai.ChatCompletion.create(messages=evidence_prompt_msg, **self.model_params)
        content = json.loads(response["choices"][0]["message"]["content"])
        evidence = content['evidence_and_explanation'] 
        
        
        processed_input["passage"] = evidence
        
        last_prompt_msg = self.last_prompt_drop_huggingface(processed_input)
        response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        
        return response
    
    def prompt_main_single_agent_e2g_hotpotqa_huggingface_llama2(self, processed_input):
        #ours
        import pdb
        # pdb.set_trace()
        try: 
            evidence_prompt_msg = self.last_prompt_hotpotqa_huggingface(processed_input)
            response = openai.ChatCompletion.create(messages=evidence_prompt_msg, **self.model_params)
            content = response["choices"][0]["message"]["content"].lower()
            try:
                json_content= json.loads(content[content.find("{"):content.find("}")+1])
                if "evidence_and_explanation" in json_content:
                    evidence = json_content["evidence_and_explanation"][0]
                else:
                    evidence = json_content["evidence and explanation"][0]

            except:
                split_answer = content.split("answer:")[1]
                if "evidence_and_explanation" in split_answer:
                    evidence = split_answer.split("evidence_and_explanation:")[1].split("\n")[0].replace("text:","").strip("{")
                elif "evidence and explanation:" in split_answer:
                    evidence = split_answer.split("evidence and explanation:")[1].split("\n")[0].replace("text:","").strip("{")  
                else:
                    print("split_answer: ", split_answer)
                    print("Evidence not here")
                    last_prompt_msg = self.cot_prompt(processed_input)
                    response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
            
            processed_input["context"]["sentences"] = [evidence] 
            last_prompt_msg = self.last_prompt_hotpotqa_huggingface(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        except:
            last_prompt_msg = self.cot_prompt(processed_input)
            response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
           
        return response
    
    def prompt_main_single_agent_e2g_hotpotqa_huggingface_palm(self, processed_input):
        prompt = self.last_prompt_hotpotqa_huggingface_palm(processed_input)
        
        
        import pdb
        # pdb.set_trace()
        
        completion = palm.generate_text(
            model=get_model(),
            prompt=prompt,
            stop_sequences=['<evidence-and-reasoning>'],
            temperature=0,
            # The maximum length of the response
            max_output_tokens=800,
        )
        print("completion.result: ", completion.result)
        # pdb.set_trace()
        try:
            evidence_explanation = json.loads(completion.result)['step-by-step-reasoning'][0]['evidence-and-reasoning']
        except Exception:
            try:
                response, evidence_explanation = completion.result.split('', maxsplit=1)
            except:
                return completion.result
                    
        # pdb.set_trace()
        print("evidence_explanation: ", evidence_explanation)
        processed_input["context"]["sentences"] = [evidence_explanation] 
        prompt = self.last_prompt_hotpotqa_huggingface_palm(processed_input)
        evidence_explanation = None
        while evidence_explanation is None:
            completion = palm.generate_text(
                model=get_model(),
                prompt=prompt,
                stop_sequences=['<evidence-and-reasoning>'],
                temperature=0,
                # The maximum length of the response
                max_output_tokens=800,
            )
        return completion
    
    

    def prompt_main_single_agent_e2g_wow(self, processed_input):
        #ours
        try: 
            # if any crashes occurs
            evidence_prompt_msg = self.last_prompt_wow(processed_input)
            response = openai.ChatCompletion.create(messages=evidence_prompt_msg, **self.model_params)
            content = json.loads(response["choices"][0]["message"]["content"])
            import pdb
            # pdb.set_trace()
            evidence = content['evidence_and_explanation']
            try:
                processed_input["context"]["sentences"] = [evidence]
            except:
                processed_input["ctxs"]=[{"text":evidence}] 
                last_prompt_msg = self.last_prompt_wow(processed_input)
                response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        except:
            return self.prompt_main_single_agent_cot_wow(processed_input)
        return response

    def prompt_main_single_agent_e2g_nq(self, processed_input):
        #ours
  
            # if any crashes occurs
        evidence_prompt_msg = self.last_prompt_nq(processed_input)
        response = openai.ChatCompletion.create(messages=evidence_prompt_msg, **self.model_params)
        content = json.loads(response["choices"][0]["message"]["content"])
        evidence = content['evidence_and_explanation']

        if content["answer"].lower() in ["cannot be determined", "unknown", "unclear", 
                                        "none", "answer not available", "none", "no information",
                                        "cannot be answered", "not available", "n/a",
                                        "not enough information", "information not provided", "it is not mentioned in the context",
                                        "not mentioned"]:
            return self.prompt_main_single_agent_cot(processed_input)
        else:
            try:
                processed_input["context"]["sentences"] = [evidence]
            except:
                if  ("w" in processed_input["question"][:3].lower() and "h" in processed_input["question"][:3].lower()) \
                or processed_input["question"].split()[0].strip("n't").lower() in \
                ["be", "am", "is", "are", "was", "were", "do", "does", "did", "have", "has", "had",\
                  "can", "could", "shall", "should", "will", "would"]:
                    processed_input["ctxs"]=[{"text":evidence}] + processed_input["ctxs"]
                else:
                     processed_input["ctxs"]=[{"text":evidence}] 
                last_prompt_msg = self.last_prompt_nq(processed_input)
                response = openai.ChatCompletion.create(messages=last_prompt_msg, **self.model_params)
        
        return response


    def prompt(self, processed_input):
        """
        OpenAI API ChatCompletion implementationn

        Arguments
        ---------
        processed_input : list
            Must be list of dictionaries, where each dictionary has two keys;
            "role" defines a role in the chat (e.g. "system", "user") and
            "content" defines the actual message for that turn

        Returns
        -------
        response : OpenAI API response
            Response from the openai python librar1y

        """
        
    
        return self.prompt_main_single_agent_e2g_logiqa_huggingface(processed_input)

        

        # prompt_main_single_agent_e2g
        # prompt_main_single_agent_cot
        # prompt_main_single_agent_direct

        # prompt_main_single_agent_evidenPrompt2gen

        # prompt_main_four_agent_coref_rank_evidence_qa
        # prompt_main_three_agent_coref_rank_qa


        # ['entity_pages']['wiki_context'], ['answer']['normalized_aliases']















