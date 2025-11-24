# SPE-LLM

This repository contains the source code of the paper *System Prompt Extraction Attacks and Defenses in Large Language Models*.

# Abstract

The system prompt in Large Language Models (LLMs) plays a pivotal role in guiding model behavior and response generation. Often containing private configuration details, user roles, and operational instructions, the system prompt has become an emerging attack target. Recent studies have shown that LLM system prompts are highly susceptible to extraction attacks through meticulously designed queries, raising significant privacy and security concerns. Despite the growing threat, there is a lack of systematic studies of system prompt extraction attacks and defenses. In this paper, we present a comprehensive framework, SPE-LLM, to systematically evaluate System Prompt Extraction attacks and defenses in LLMs. First, we design a set of novel adversarial queries that effectively extract system prompts in state-of-the-art (SOTA) LLMs, demonstrating the severe risks of LLM system prompt extraction attacks. Second, we propose three defense techniques to mitigate system prompt extraction attacks in LLMs, providing practical solutions for secure LLM deployments. Third, we introduce a set of rigorous evaluation metrics to accurately quantify the severity of system prompt extraction attacks in LLMs and conduct comprehensive experiments across multiple benchmark datasets, which validates the efficacy of our proposed SPE-LLM framework.

Paper Link: https://arxiv.org/pdf/2505.23817

# Framework Overview
![CHEESE!](FrameworkOverview)

# How to use the code

1. Download the required datasets from the sources mentioed in the paper.
2. To perform attack on open-sourced models, run the main_attack.py with the seleted model, dataset, and promting technique.
3. To perform defense on open-sourced models, run the main_defense.py, uncommenting the corresponding defenses within *get_response* function, with the seleted model, dataset, and promting technique.
4. To perform attack/defense on close-sourced models, run the notebook with the seleted model, dataset, and promting technique.
5. Make sure to use your own API keys. 

