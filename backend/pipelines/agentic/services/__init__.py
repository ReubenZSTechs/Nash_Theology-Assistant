"""
The building blocks of Graph-of-Agents, one file per step of the algorithm:

    node_sampling_service.py     1. choose the k most suitable agents
    response_service.py          2. every chosen agent answers on its own
    edge_sampling_service.py     3. agents score each other -> relevance + directed edges
    message_passing_service.py   4. source -> target, then target -> source refinement
    graph_pooling_service.py     5. combine the answers (mean or max pooling)

    agent_pool.py                shared helpers: load agents, call them in parallel, format answers
    prompts.py                   every user-prompt template in one place
"""
