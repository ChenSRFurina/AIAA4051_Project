"""Typeset the six-page Project 5 paper from the released observations."""
from render_final_project import E, percent, tbl, save


STYLE = """
*{box-sizing:border-box}html{font-family:'Segoe UI',Arial,sans-serif;color:#18342f;background:#dadfd9}
body{margin:0}a{color:#155f55}.sheet{position:relative;width:210mm;height:297mm;margin:6mm auto;background:#fffefa;padding:15mm 17mm 17mm;break-after:page}
.sheet:last-child{break-after:auto}.folio{position:absolute;bottom:9mm;left:17mm;right:17mm;border-top:.3mm solid #a9b7ae;padding-top:2mm;display:flex;justify-content:space-between}
.mast,.folio,.tag{font-size:8.4pt;letter-spacing:.1em;font-weight:650;color:#486a5d;text-transform:uppercase}
.mast{border-bottom:1mm solid #245e50;padding-bottom:2.5mm;margin-bottom:6.5mm}
h1,h2{font-family:Georgia,'Times New Roman',serif;font-weight:400;letter-spacing:-.035em;color:#173d34;line-height:1.07}
h1{font-size:35pt;margin:1.5mm 0 4mm}h2{font-size:26pt;margin:0 0 5mm}h3{font-size:12pt;margin:4.7mm 0 1.4mm;color:#174b42}p{font-size:10.25pt;line-height:1.42;margin:0 0 2.4mm}
.lead{font:15pt/1.33 Georgia,'Times New Roman',serif;margin:2mm 0 4mm}.tag{display:block;font-size:8pt;margin:2mm 0}
.callout{background:#eaf0e8;border-left:1.1mm solid #33755e;padding:3mm 4mm;margin:3.6mm 0}.callout p:last-child{margin:0}
table{border-collapse:collapse;width:100%;font-size:9.4pt;line-height:1.22;font-variant-numeric:tabular-nums;margin:1.6mm 0 2.2mm}th,td{text-align:left;vertical-align:top;padding:1.65mm 1.6mm;border-bottom:.2mm solid #c6d1c7}th{color:#264c41;background:#e6eee6;font-weight:650}
.compact table{font-size:8.6pt}.compact th,.compact td{padding:1.15mm 1.5mm}.note,.refs p{color:#47615a;font-size:8.9pt;line-height:1.35}.big{display:grid;grid-template-columns:1fr 1fr;gap:4mm;margin:4mm 0}.big>div{background:#eaf0e8;padding:4mm}
.big strong{font:25pt Georgia;color:#245e50;display:block}.route{border:1px solid #b5c8bb;padding:2.7mm 3mm;margin:2mm 0;font-size:10pt;line-height:1.4}
p{font-size:11.25pt;line-height:1.47}h3{font-size:13.1pt}.compact table{font-size:9.2pt}.compact th,.compact td{padding:1.45mm 1.5mm}.note,.refs p{font-size:9.4pt}
.paperlinks{display:inline}.paperfiles{display:none}@page{size:A4;margin:0}@media print{html{background:white}body{margin:0}.sheet{margin:0;box-shadow:none}.paperlinks{display:none}.paperfiles{display:inline}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
@media screen and (max-width:799px){.sheet{max-width:100%;width:auto;height:auto;margin:0 auto 8px;padding:20px 15px 30px}.folio{position:static;margin-top:22px}.scroll{overflow-x:auto}}
"""

def paragraphs(*lines):
    return ''.join('<p>'+line+'</p>' for line in lines)


def section(title,*lines):
    return '<h3>'+title+'</h3>'+paragraphs(*lines)


def paper(s):
    ds=s["datasets"]
    hp,sq=ds["hotpotqa"],ds["searchqa"]
    pages=[]
    def page(label,body):
        num=len(pages)+1
        pages.append(f'<section class="sheet"><div class="mast">AIAA 4051 &nbsp; / &nbsp; NLP Research Project 5 &nbsp; · &nbsp; {E(label)}</div>{body}<footer class="folio"><span>Graph-structured skills for question answering</span><span>{num} / 6</span></footer></section>')
    page("Research question", '<span class="tag">Fall 2026 &nbsp; · &nbsp; 25 November 2026</span><h1>Can a successful answer help a skill learn from failure?</h1>'
         +'<p class="lead">A controlled reproduction of GraphSkillEvo on SearchQA and HotpotQA with training-only contrastive feedback.</p>'
         + section('Abstract','An agent skill describes how to route a question and which steps to execute. We tested whether adding a successful training example to the failure evidence used to revise a skill improves held-out question answering. We fixed the model, failure sampler, selection rule and inference setting. On a single local 4-bit Qwen3.5-9B, the original paired-feedback optimizer (C) tied the baseline optimizer (B) on 1,000 HotpotQA questions and lost two exact matches on 50 SearchQA questions. Under common call and token ceilings, HotpotQA selected the same skill for both arms. A later pairing variant improved HotpotQA but lost ground on SearchQA.')
         +'<div class="big"><div><strong>20 / 20</strong>HotpotQA repairs / regressions<br>Original C versus B on test</div><div><strong>0 / 2</strong>SearchQA repairs / regressions<br>Original C versus B on test</div></div>'
         +section('Why include successes?','GraphSkillEvo [1] edits reusable instructions using failed attempts. Failures reveal where a procedure breaks, but provide little evidence about which behavior to preserve. Inspired by the experience memory in JitRL [4], we retrieved prior successes from this optimizer’s own training runs. The solver then used the frozen skill alone. Neither weights nor decoding logits were updated.')
         +section('What the experiment measures','The primary comparison is the pre-existing B versus original C design. N and I measure what happens without optimization. C2b and C2c examine feedback wording and retrieval. We compare predictions by original question ID and report both fixes and new errors.'))
    page("Experimental design",'<h2>Data, selection and scoring</h2>'+
         tbl(['Dataset','Train / dev / test','Source and selection'],[['SearchQA [2]','100 / 30 / 50','The published GraphSkillEvo ID split, preserving its partitions.'],['HotpotQA [3]','300 / 150 / 1,000','Course update of 2 November. Distractor train v1.1 and official distractor dev v1.']])
         +paragraphs('For HotpotQA we sorted original IDs within each training difficulty, seeded the sampler with 42, and drew 150 without replacement per stratum. The first 100 entered train and the next 50 entered dev. The 1,000 project test questions were sampled from the official distractor development set after excluding 30 previously examined IDs. The official hidden test set was not used. All three ID lists are disjoint. The pinned Hugging Face dataset revision starts with <b>1908d6af</b>.',
                     'HotpotQA supplied every distractor paragraph. SearchQA used supplied snippets with a 4,096-token context allowance. The solver saw question, evidence and optional skill. Gold answers and supporting-fact labels were withheld. All conditions used the same local Q4_K_M weights, 16,384-token server window and thinking disabled.')
         +section('How a generation works','Two initial candidates share their initialization across evolutionary conditions. One is the upstream initial skill and one is model-generated. Train rollouts supply successes, failures and feedback. Each of three rounds uses the same train batch across arms (20 SearchQA or 100 HotpotQA questions), samples up to three failures per parent feedback, and proposes at most two new candidates. The two operators edit global guidance and graph structure respectively.',
                 'Dev EM then F1 order candidates for rank-weighted parent sampling. Survivors are ranked by dev EM with ties favoring existing parents and stable order. The final skill is the best survivor by dev EM. Repeated selection on the same small dev set can favor incidental gains. Test outputs were reserved for scoring and diagnosis after skill selection.')
         +section('Execution and measurement','QA uses temperature 0, up to 2,048 generated tokens. Operators use temperature 0.6 and up to 3,072 tokens. We extract the last complete answer tag, falling back to the last nonempty line. We lowercase, remove ASCII punctuation and English articles, and collapse whitespace. EM compares normalized strings. F1 measures multiset token overlap, taking the best reference. HotpotQA enforces the incompatible yes/no/noanswer rule. Reported latency is mean measured seconds per question.'))
    page("Part 1 · Reproduction",'<h2>Where skill evolution helped and hurt</h2>'+
         tbl(['Validation-selected B','Initialization','Round 1','Round 2','Round 3'],[['SearchQA · correct / 30','23','23','24','24'],['HotpotQA · correct / 150','102','104','105','105']])
         +section('HotpotQA acquired two routes with the same execution steps','Initialization retained the upstream skill with no explicit routes or nodes. Its model-generated sibling had lookup, multi-hop and contradiction routes but scored 97/150. Round 1 changed global guidance and reached 104. In round 2, a graph mutation produced the selected skill below. Both round 3 children lost dev accuracy.')
         +'<div class="route"><strong>Round 2 routing</strong><br>Standard Extraction Task &nbsp; ┐<br>Entity Relationship Task &nbsp; ┘ &nbsp;→ identify contextual entities → locate specific attribute → extract exact answer → verify against context</div>'
         +paragraphs('Compared with its parent, the round 2 child fixed four answers and broke three. It narrowed “Mitch Rapp series” to the reference “Mitch Rapp”, but replaced the correct “grandfather” with “father” for the relation of “Slew” McCain to the presidential candidate. Both routes direct the solver to verify the answer, yet neither requires it to trace each generation. This missing check is one plausible explanation. The graph records instructions, not the model’s internal execution.')
         +section('SearchQA gained one dev answer between rounds 1 and 2','The generated initializer supplied Direct Fact Retrieval, Multi-Step Reasoning and Information Not Found. The round 2 winner kept those routes and changed the global guidance. Both round 1 children tied their parent. The last round supplied no new best. The appendix contains the task conditions, ordered nodes, complete text and parent-relative outcomes of all recorded candidates.')
         +section('Static baselines expose answer-boundary changes','On the Gershwin and Rachmaninoff clue, SearchQA N and I output “Rhapsody”, which matches “a rhapsody” after normalization. B gave “Rhapsody in Blue”, one work instead of the requested form. On a HotpotQA birthday, N and B gave the reference “3 August 1980”, while I reordered it to “August 3, 1980” and lost EM. Such cases motivate reporting token F1 alongside exact matches.'))
    score_rows=[]
    for a,label in [('N','No skill'),('I','Initial skill'),('B','Failure feedback'),('C','Paired feedback'),('C2b','Evidence diagnostics'),('C2c','Clue-aware retrieval')]:
        score_rows.append([a+' · '+label]+[v for d in (sq,hp) for v in (d['arms'][a]['dev']['correct'],f"{100*d['arms'][a]['test']['em']:.1f}",percent(d['arms'][a]['test']['f1']),f"{d['arms'][a]['test']['mean_latency_seconds']:.2f}")])
    budgetrows=[]
    for name,d in [('SearchQA',sq),('HotpotQA',hp)]:
        for a,m in d['matched'].items():
            c=m['checkpoint']; budgetrows.append([name+' '+a,c['round'],f"{c['cost']['calls']:,} / {c['cost']['generated_tokens']:,}",f"{d['caps']['calls']:,} / {d['caps']['tokens']:,}",percent(m['metrics']['em'])+' / '+percent(m['metrics']['f1'])])
    page('Part 3 · Comparisons','<h2>One fixed model, six conditions</h2><p>Table 1 gives dev correct, test EM and F1 (%) and mean test latency (s). SearchQA (SQ) has 50 test questions and HotpotQA (HP) has 1,000. The appendix includes counts and latency percentiles.</p>'
         +'<div class="compact">'+tbl(['Condition','SQ dev /30','SQ EM','SQ F1','SQ s','HP dev /150','HP EM','HP F1','HP s'],score_rows)+'</div>'
         +paragraphs('I led SearchQA (39/50). C2c led HotpotQA (639/1,000), six above N. SearchQA C2b and C2c produced identical frozen text and share a single test observation. Scores depend on both the dataset and the output form.')
         +section('Primary comparison under common resource ceilings','For each dataset take the smaller B/C final development totals separately for logical calls and generated tokens. Pick the last complete round satisfying both limits. A round is indivisible. This leaves different amounts of budget unused. Each selected checkpoint has a held-out score.')
         +'<div class="compact">'+tbl(['Checkpoint','Round','Used calls / tokens','Common ceilings','Test EM / F1 (%)'],budgetrows)+'</div>'
         +paragraphs('For HotpotQA the matched C and B have identical skill hashes, and every rendered prompt matches the observed B test prompt. Both therefore reuse B’s 1,000 answers. Full-round C selected a different skill and scored 61.8 EM and 78.41 F1. SearchQA matched checkpoints retained their respective final skills. The SearchQA rule was recorded before local testing. HotpotQA matching applies the existing cost rule retrospectively.')
         +section('How many answers changed?','At full rounds, HotpotQA C repaired 20 of B’s 382 errors and lost 20 of B’s 618 hits. The paired 95% bootstrap interval for the EM difference was −1.3 to +1.3 percentage points. SearchQA C repaired 0/15 and lost 2/35 (−4 points, interval −10 to 0). HotpotQA C2c repaired 55 and lost 34 (+2.1 points, interval +0.2 to +3.9). SearchQA C2c repaired one and lost six (−10 points, interval −20 to 0). Intervals resample questions within this run.'))
    order=['S','G','T','L','A','V','R','C','F','U']
    counts=s['failure_counts']
    rows=[[s['taxonomy'][k][0]]+[counts[d][sp].get(k,0) for d in ('searchqa','hotpotqa') for sp in ('dev','test')] for k in order]
    page('Part 2 · Failure analysis','<h2>What the baseline missed</h2>'+paragraphs('We reviewed every EM-zero observation of the selected B skill. Development failures informed hypotheses about feedback. Test annotations describe frozen outputs. Each entry includes the original ID, question, references, prediction, score, diagnosis, raw response and a literal context excerpt.')
         +'<div class="compact">'+tbl(['Primary label','SearchQA dev','SearchQA test','HotpotQA dev','HotpotQA test'],rows+[['All EM-zero',6,15,45,382]])+'</div>'
         +section('Three different reasons for zero EM','<b>Answer boundary.</b> For HotpotQA many responses describe a plausible referent in a different form. The largest test group (212/382) includes possible aliases, inflections and span boundaries. Membership does not imply semantic correctness. An exact-span check can help with an extra qualifier but can also drop information required by the question.',
                 '<b>Requested target.</b> In the SearchQA clue “1940: Bad boys are turned into donkeys”, B gave “Pleasure Island” for “Pinocchio”. The response identifies a setting associated with the story. The final skill’s generic multi-step route does not explicitly distinguish a film title from a location. The proposed cause is an interpretation of the clue and the adopted text.',
                 '<b>Question or reference concerns.</b> The HotpotQA Rose/Millettia question presupposes a common family. B challenged the premise, while the reference says “Fabaceae”. We flag the tension and preserve the official score. We did the same for under-specified clues when evidence did not establish the intended slot.')
         +'<p class="note">One primary label per failure. We prioritize an observable format problem, explicit abstention or concrete target mismatch where supported. A different value alone leaves the failed reasoning step uncertain. The released labels are analyst-assisted and have no independent second-rater agreement estimate.</p>')
    exposure=[]
    for name,d in [('SearchQA',sq),('HotpotQA',hp)]:
        vals=[]
        for a in ('C','C2b','C2c'):
            p=d['pairing'][a]; vals.append(f"{p['selected']-p.get('unpaired',0)}/{p['selected']}")
        exposure.append([name]+vals)
    page('Interpretation · Reproduction','<h2>From a training pair to a held-out claim</h2>'
         +section('What the optimizer received','C looks for a previous success on the same training ID and visible context, then searches similar questions by type and lexical overlap. If no success can be paired, it falls back to baseline feedback. C2b adds observable diagnostics, literal evidence and scope instructions. C2c also uses clues to the requested answer type. The failure sampler is unchanged. Coverage counts failure exposures in the feedback sent to scheduled mutations, including repeated parents.')
         +'<div class="compact">'+tbl(['Paired / selected exposures','C','C2b','C2c'],exposure)+'</div>'
         +paragraphs('One HotpotQA C2c parent answered “songwriters” for the training reference “songwriter”. A retrieved success answered “genetics” to a different question. The child adopted a global rule about matching answer form. On dev it changed “operas” to “opera” and “2” to “two”, and regressed from “grandfather” to “father”. The added rule explicitly asks to match gold and mentions capitalization. The deployed solver cannot see gold, and the scorer ignores case. Several instructions changed at once, limiting attribution.')
         +section('Cost, uncertainty and conclusion',f'Final local experiment ledgers contain 13,952 physical calls, 30,973,967 prompt tokens and 1,685,589 generated tokens across the two datasets, with zero failed requests. Local GPU allocation from all recorded start/stop spans, including pilots, was {s["gpu"]["local_hours"]:.3f} hours on one RTX 4070 Ti SUPER. The course specifies an A6000 and a proposed 40-hour cap. Historical teammate-API GPU hours are unknown, so total team GPU allocation cannot be certified.',
                 'One seed, repeated validation selection and 50 SearchQA test items limit generalization. SearchQA test IDs had been evaluated on a previous hosted model. Both test sets are now exposed. Original C supplied no mean EM gain here. Clue-aware feedback improved HotpotQA relative to B but scored below B on SearchQA. The evidence supports a dataset-specific effect. An independent split and multiple seeds would test its stability.')
         +'<div class="refs"><h3>References and evidence</h3><p>[1] <a href="https://arxiv.org/abs/2609.21749">GraphSkillEvo</a>, 2026. Upstream code revision 722a8a9. [2] Dunn et al., <a href="https://arxiv.org/abs/1704.05179">SearchQA</a>, 2017. [3] Yang et al., <a href="https://aclanthology.org/D18-1259/">HotpotQA</a>, EMNLP 2018. [4] Li et al., <a href="https://arxiv.org/abs/2601.18510">Just-in-Time Reinforcement Learning</a>, 2026.</p><p>In the extracted submission, open <span class="paperfiles">appendix.html and demo.html</span><span class="paperlinks"><a href="appendix.html">appendix.html</a> and <a href="demo.html">demo.html</a></span> alongside this paper for the evidence and demonstration.</p></div>')
    save('report.html','<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Can a successful answer help a skill learn from failure?</title><style>'+STYLE+'</style></head><body>'+''.join(pages)+'</body></html>')
