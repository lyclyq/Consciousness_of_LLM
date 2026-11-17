r"""11-10 (user: KL as a relation, not a threshold): the Chinese counterparts of fix_v5_kl_relation_en_20261110.py and the English hand
edits of the abstract, introduction, contribution 4, methods, discussion and conclusion. CRLF kept. Old strings must match once.
usage: fix_v5_kl_relation_zh_20261110.py PAPER_SOURCE"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / 'self_zh_body.tex'
t = p.read_text(encoding='utf-8')
R = [
    ('即使没有任何问题，模型在自己消息上对下一个 token 的预测，也会把这条消息与另外两种角色下的同一文字区分开，13 个有自我索引的模型中有 8 个如此，12 个没有自我索引的模型中一个也没有，而且同样的坐标承载着这一差别。',
     '即使没有任何问题，模型在自己消息上对下一个 token 的预测，也与另外两种角色下同一文字的预测不同，消息上带有自我索引的层越多，差别越大（二十五个模型上 Spearman \\(\\rho = 0.82\\)），而且同样的坐标承载着这一差别。'),
    ('在消息自身的 token 上，13 个有自我索引的模型中有 8 个对自己消息的下一个 token 的预测，不同于对另外两种角色下同一文字的预测，12 个没有自我索引的模型中一个也没有（第~\\ref{sec:res_nexttoken} 节）。',
     '在消息自身的 token 上，模型对自己消息的下一个 token 的预测不同于对另外两种角色下同一文字的预测，消息上带有自我索引的层所占比例越大，这一差别越大（二十五个模型上 Spearman \\(\\rho = 0.82\\)，自然顺序下为 0.87，第~\\ref{sec:res_nexttoken} 节）。'),
    ('即使没有任何问题，对消息上下一个 token 的预测也会把模型自己的消息与两个外部消息区分开，13 个有自我索引的模型中有 8 个如此，12 个没有自我索引的模型中一个也没有，在同样的置换检验下这种格局出现的概率为 0.001，而且在全部五种读出中，索引坐标承载着这一差别。',
     '即使没有任何问题，消息上带有自我索引的层越多，对消息上下一个 token 的预测就越把模型自己的消息与外部消息区分开（Spearman \\(\\rho = 0.82\\)），而且在全部五种读出中，索引坐标承载着这一差别。'),
    ('两个区间的下端都大于 1，就算模型在下一个 token 预测中把自己的消息单独分出。',
     '我们用两个比值中较小的一个概括一个模型。比值为 1 表示模型自己的消息与外部消息之间的差别，和两个外部消息彼此之间的差别一样大。'),
    ('在消息自身的 token 上，13 个有自我索引的模型中有 8 个对自己消息的下一个 token 的预测，把它与另外两种角色下的同一文字区分开，12 个没有自我索引的模型中一个也没有（第~\\ref{sec:res_nexttoken} 节）。',
     '在消息自身的 token 上，模型对自己消息与另外两种角色下同一文字的下一个 token 预测不同，消息上带有自我索引的层越多，差别越大（二十五个模型上 Spearman \\(\\rho = 0.82\\)，第~\\ref{sec:res_nexttoken} 节）。'),
    ('即使没有任何问题，消息上下一个 token 的预测也会在 13 个有自我索引的模型中的 8 个里把模型自己的消息区分开，在 12 个没有自我索引的模型中一个也没有，而且索引坐标承载着这一差别：',
     '即使没有任何问题，消息上带有自我索引的层越多，消息上下一个 token 的预测就越把模型自己的消息与外部消息区分开（\\(\\rho = 0.82\\)），而且索引坐标承载着这一差别：'),
]
for old, new in R:
    assert t.count(old) == 1, old[:30]
    t = t.replace(old, new)
# discussion: "aligned with u in most models that carry the index"
for old in ('在多数带有索引的模型中', '在大多数带有索引的模型中'):
    if t.count(old) == 1:
        t = t.replace(old, '按索引在消息上的覆盖程度')
        print('discussion phrase replaced:', old)
        break
else:
    print('WARNING: discussion phrase not found')
# results subsection
i = t.index('归属问题问的是来源。没有问题询问来源时')
j = t.index('\\section{', i)
NEW = r"""归属问题问的是来源。没有问题询问来源时，模型同样在处理自己的消息：在消息的每个 token 上，它都预测下一个 token。我们比较同一份留出正文三种角色下的这些内容预测，用 \(\mathrm{KL}(A\Vert U)\) 与 \(\mathrm{KL}(A\Vert S)\) 对两个外部来源之间的散度 \(\mathrm{KL}(U\Vert S)\) 之比中较小的一个来概括（第~\ref{sec:setup_nexttoken} 节，附录~\ref{app:nexttoken}）。

在主探针下，这个比值随模型自己的消息在内容上被单独分出的层所占比例而增大（图~\ref{fig:self_kl}a，表~\ref{tab:self_kl}），二十五个模型上的 Spearman 相关为 \(\rho = 0.82\)，在十三个有自我索引的模型中为 0.78。在 69\%--100\% 的层上于消息内容上带有索引的 8 个模型，比值在 2.0 到 65 之间：GLM-4.7-Flash（2.0）、phi-4（2.3）、Qwen3.8-27B（2.7）、MiMo-V2.6-Distill-Qwen-9B（5.2）、gpt-oss-120b（9.7）、EXAONE-4.0.1-32B（12.8）、gpt-oss-20b（35）与 GLM-4.6V-Flash（65）。没有自我索引的模型，以及只在不到一半的层上于内容上带有索引的 DeepSeek-R1-Distill-Llama-70B 与两个 Granite 模型，比值在 0.05 到 1.11 之间，即它们自己的消息与外部消息的差别，至多与两个外部消息彼此之间的差别相当。Qwen3.8-Flash-Next 与 GLM-5.3-Flash 的内容覆盖是四条残差流的平均，它们的比值为 1.01 与 1.13。

在自然对话顺序下，关系相同（图~\ref{fig:self_kl}b，表~\ref{tab:self_kl_natural}），十三个模型上 \(\rho = 0.87\)。在超过一半的层上于消息内容上带有索引的模型，比值在 1.6 到 5.9 之间，只在 5 层或更少的层上这样做的模型在 0.14 到 0.72 之间。MiMo-7B-RL 在主探针下没有自我索引，在自然顺序下则在其 36 层中的 34 层上有，它的比值从 0.57 升到 2.0。只接受自然顺序、几乎每一层都有自我索引的 DeepSeek-V4.1-Flash 比值为 2.4，没有自我索引的 Ministral-3-14B 两个版本的比值为 0.14 与 0.29。在一个模型之内，这种区分也随自我索引变化。去掉模板加在模型自己消息上的空思考块会削弱 Qwen3.8-27B 的自我索引（第~\ref{sec:res_inducer} 节），此时不思考模型的完整比值从 308 降到 1.4（区间 0.8 至 3.0）。

在 Qwen 模型中，完整散度的大部分落在控制 token 上。在 Qwen3.8-27B 中，User 消息的正文几乎在每个位置上都预测本轮结束（不思考时为 0.99），去掉控制 token 后，比值在不思考时从 308 降到 4.0，在思考时从 40 降到 8.5。没有自我索引的 Qwen3.6-27B 只凭这些 token 就有 39 的完整比值，在内容上则为 0.33。在 gpt-oss 模型、MiMo-V2.6-Distill-Qwen-9B 和其他模型中，控制 token 在正文上几乎没有概率，差别落在内容的预测上。

这种差别跟随第一层，而不是第二层。GLM-4.6V-Flash、GLM-4.7-Flash 与 phi-4 有自我索引却不认出自己的消息，它们的比值（2.0 到 65）与认出自己消息的模型处在同一范围。推理之后能认出自己消息的 Qwen3.8-Flash-Next 与 GLM-5.3-Flash，比值为 1.0 与 1.1。因此，对下一个 token 的预测在消息本身上读取索引，先于任何问题，也与问题无关，消息上带有索引的层越多，读取越强，无论模型是否认出自己的消息。第~\ref{sec:res_mechanism} 节表明，索引的坐标承载着这一差别。

"""
t = t[:i] + NEW + t[j:]
p.write_text(t, encoding='utf-8', newline='\r\n')
print('ok')
