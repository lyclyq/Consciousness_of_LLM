"""The seven models added on 2026-11-03/04 (five developers new to the comparison), in one place for every analysis script.

Each entry: display name, developer, language of the items, the runs (relative to runs/<group>/ of the wave-3 archive), and the
modes the chat template provides. Measurement as for the earlier models of the same kind:
  non_thinking  suite G with the non-thinking rendering (geometry and native letter probabilities), suite M
  thinking      thinking attribution (150 bodies x 3 roles x 2 orders, greedy, 3072 new tokens) and thinking M; geometry under the
                thinking rendering where the template has no non-thinking mode (thinking only), or where the model shows thinking
                self-reference (the rule of Qwen3.5-27B: thinking-rendering geometry only then)
  natural       crossmodel_natural3 (three roles) or crossmodel_natural (Assistant and User; templates that refuse a System
                message after a user turn), only for the family's smallest model in the comparison
MiMo-V2.6-Distill-Qwen-9B: classified by its template's default call, which equals enable_thinking=True of the model card; under
greedy decoding the model closes the reasoning block at once (median 4 new tokens), stated in the appendix.
"""

NEWDEV = {
    'ernie45_21b': dict(name='ERNIE-4.5-21B-A3B-Thinking', short='ERNIE-4.5-21B-A3B', developer='Baidu', lang='zh', modes=('thinking',),
                        geo={'thinking': 'suite/ernie45_21b_a3b_thinking'}, attr=None, think='thinking/ernie45_21b_a3b_thinking',
                        m=None, thinkm='thinking_m/ernie45_21b_a3b_thinking',
                        natural=('natural3', 'suite_nat3/ernie45_21b_a3b_thinking'), color='#3b5bdb'),
    'mimo7b_rl': dict(name='MiMo-7B-RL-0530', short='MiMo-7B-RL', developer='Xiaomi', lang='zh', modes=('non_thinking', 'thinking'),
                      geo={'non_thinking': 'suite/mimo7b_rl_0530'}, attr='suite/mimo7b_rl_0530', think='thinking/mimo7b_rl_0530',
                      m='suite/mimo7b_rl_0530_M', thinkm='thinking_m/mimo7b_rl_0530',
                      natural=('natural3', 'suite_nat3/mimo7b_rl_0530'), color='#f76707'),
    'mimo_vl7b': dict(name='MiMo-VL-7B-RL-2508', short='MiMo-VL-7B', developer='Xiaomi', lang='zh', modes=('thinking',),
                      geo={'thinking': 'suite/mimo_vl7b_rl_2508'}, attr=None, think='thinking/mimo_vl7b_rl_2508',
                      m=None, thinkm='thinking_m/mimo_vl7b_rl_2508', natural=None, color='#fd7e14'),
    'mimo_distill9b': dict(name='MiMo-V2.6-Distill-Qwen-9B', short='MiMo-V2.6-Distill-9B', developer='Xiaomi', lang='zh',
                           modes=('non_thinking', 'thinking'),
                           geo={'non_thinking': 'suite/mimo_v26_distill_qwen9b', 'thinking': 'suite_think/mimo_v26_distill_qwen9b'},
                           attr='suite/mimo_v26_distill_qwen9b', think='thinking/mimo_v26_distill_qwen9b',
                           m='suite/mimo_v26_distill_qwen9b_M', thinkm='thinking_m/mimo_v26_distill_qwen9b', natural=None,
                           color='#c2255c'),
    'moonlight16b': dict(name='Moonlight-16B-A3B-Instruct', short='Moonlight-16B-A3B', developer='Moonshot AI', lang='zh',
                         modes=('non_thinking',), geo={'non_thinking': 'suite/moonlight16b_a3b_instruct'},
                         attr='suite/moonlight16b_a3b_instruct', think=None, m='suite/moonlight16b_a3b_instruct_M', thinkm=None,
                         natural=('natural3', 'suite_nat3/moonlight16b_a3b_instruct'), color='#495057'),
    'apertus8b': dict(name='Apertus-8B-Instruct-2509', short='Apertus-8B', developer='Swiss AI', lang='en', modes=('non_thinking',),
                      geo={'non_thinking': 'suite/apertus8b_instruct_2509_en'}, attr='suite/apertus8b_instruct_2509_en', think=None,
                      m='suite/apertus8b_instruct_2509_en_M', thinkm=None,
                      natural=('natural2', 'natural/apertus8b_instruct_2509_en'), color='#2f9e44'),
    'phi4': dict(name='phi-4', short='phi-4', developer='Microsoft', lang='en', modes=('non_thinking',),
                 geo={'non_thinking': 'suite/phi4_en'}, attr='suite/phi4_en', think=None, m='suite/phi4_en_M', thinkm=None,
                 natural=('natural3', 'suite_nat3/phi4_en'), color='#1971c2'),
}
