#!/bin/bash
# PRO, user 10-26: gpt-oss-120b, the 2x2 of the causal tests (the source matters: attribution question, secret game v5; it does not:
# content question, arithmetic question), every layer, rank 8, with the protocol of gpt-oss-20b (thinking rendering, English bodies,
# native MXFP4 on one GPU, prompt-only edits), then the 64-dimension attribution diagnostic. No thirds and no position groups (user
# 10-26: the layer-window maps replace the thirds). Runs in tmux session g120. Skip with STOP_GPTOSS120_2X2 (checked before every step).
#   0. waits until gptoss_games5_pro.sh (the gpt-oss-20b secret game v5) has exited
#   1. fit on the thinking rendering (crossmodel_thinking_ablate2.py, the 75 discovery bodies, k up to 64), as for gpt-oss-20b
#   2. smokes: attribution (first 4 held-out bodies x 2 orders, the six Assistant/User conditions) and secret game v5 (8 games, main
#      question and its control)
#   3. release rule fixed in advance: attribution, unedited own >= 0.5 and User <= 0.25 answered "you" on finished items, >= 75%
#      finished; game, as for the other readouts (setter's role >= 0.5 own and <= 0.25 user on finished games, >= 75% finished, control
#      correct in >= 7 of 8 games under both unedited conditions). A failed smoke stops its branch and is left for inspection.
#   4. full runs: attribution (the ten conditions of the gpt-oss-20b run, System included), content and arithmetic questions (the six
#      conditions of gpt-oss-20b), secret game v5 main and control (six conditions)
#   5. 64-dimension attribution diagnostic (unedited items copied from step 4, identical prompts)
C=/root/p2_scale_20261015/crossmodel; PY=/root/consciousness_a6000pro/.venv/bin/python; M=/root/hf_models/openai/gpt-oss-120b
B=/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl
mkdir "$C/gptoss120_2x2.lock" 2>/dev/null || exit 0
cd $C
st() { echo "$(date '+%F %T') $1" >> gptoss120_2x2_status.log; }
stop() { [ -e STOP_GPTOSS120_2X2 ] && { st "stopped by STOP_GPTOSS120_2X2"; exit 0; }; }
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONPATH=/root/p2_scale_20261015/tf_next PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120
FIT=thinking_ablate4/gpt_oss_120b_fit/fit.pt
LOAD="--model $M --on-kwargs {} --lang en --batch 8 --max-new 3072 --max-memory-gib 90 --save-gen"
SW="--mode thinking $LOAD --bodies bodies_en_150_list.json --splits $B --fit-from $FIT --band 0-35"
C6="base_A:assistant:-:-:0,base_U:user:-:-:0,swU8_A:assistant:user:V:8,swA8_U:user:assistant:V:8,rswU8_A:assistant:user:R:8,rswA8_U:user:assistant:R:8"
ALL="base_A:assistant:-:-:0,base_U:user:-:-:0,base_S:system:-:-:0,idA8_A:assistant:assistant:V:8,swU8_A:assistant:user:V:8,swS8_A:assistant:system:V:8,rswU8_A:assistant:user:R:8,swA8_U:user:assistant:V:8,rswA8_U:user:assistant:R:8,swA8_S:system:assistant:V:8"
CON="base_A:assistant:-:-:0,swU8_A:assistant:user:V:8,swS8_A:assistant:system:V:8,rswU8_A:assistant:user:R:8,base_U:user:-:-:0,swA8_U:user:assistant:V:8"
K64="base_A:assistant:-:-:0,base_U:user:-:-:0,swU64_A:assistant:user:V:64,swA64_U:user:assistant:V:64,rswU64_A:assistant:user:R:64,rswA64_U:user:assistant:R:64"
G="crossmodel_games.py --mode thinking $LOAD --fit-from $FIT --band 0-35 --items games/games_en_secret5.json --games secret --conds $C6"
until ! pgrep -f '[g]ptoss_games5_pro.sh' > /dev/null; do sleep 60; done
stop
st "0 start (role_swap $(sha256sum crossmodel_role_swap.py | cut -c1-8), games $(sha256sum crossmodel_games.py | cut -c1-8), items $(sha256sum games/games_en_secret5.json | cut -c1-8))"
if [ ! -s $FIT ]; then
  st "1 fit start"
  $PY crossmodel_thinking_ablate2.py --suite smoke --layers band --no-gen-edit --edit-end before_gen --main-layer 23 --band 0-35 --limit-bodies 1 \
    --conds base_A:assistant:-:0 $LOAD --bodies bodies_en_150_list.json --splits $B --out thinking_ablate4/gpt_oss_120b_fit > gptoss120b_fit.log 2>&1
  st "1 fit exit $? ($(ls -la $FIT 2>/dev/null | awk '{print $5}') bytes)"
fi
[ -s $FIT ] || { st "1 fit FAILED"; exit 1; }
stop
st "2 smokes start"
$PY crossmodel_role_swap.py $SW --conds $C6 --limit-bodies 4 --out thinking_swap/gpt_oss_120b_2x2_smoke > gptoss120b_attr_smoke.log 2>&1
for q in main control; do
  $PY $G --qkind $q --limit-bodies 8 --out games_swap/gpt_oss_120b_s5_smoke_$q > gptoss120b_g5_smoke_$q.log 2>&1
done
verdict=$($PY - <<'EOF'
import glob, json
its = [json.load(open(f)) for f in glob.glob('thinking_swap/gpt_oss_120b_2x2_smoke/items/*.json')]
def share(c):
    r = [x for x in its if x['cond'] == c]
    f = [x for x in r if x.get('finished', True)]
    return (sum(x.get('on_answer') == 'self' for x in f) / max(1, len(f)), len(f), len(r))
a, u = share('base_A'), share('base_U')
ok_a = a[2] > 0 and u[2] > 0 and a[1] >= 0.75 * a[2] and u[1] >= 0.75 * u[2] and a[0] >= 0.5 and u[0] <= 0.25
m = json.load(open('games_swap/gpt_oss_120b_s5_smoke_main/summary.json'))['secret']
c = json.load(open('games_swap/gpt_oss_120b_s5_smoke_control/summary.json'))['secret']
ga, gu = m['base_A'], m['base_U']
ok_g = (ga['finished'] >= 0.75 * ga['n'] and gu['finished'] >= 0.75 * gu['n'] and ga['own_share'] >= 0.5 and gu['own_share'] <= 0.25
        and all(round(c[k]['accuracy'] * c[k]['finished']) >= 7 for k in ('base_A', 'base_U')))
print(('PASS' if ok_a else 'FAIL'), ('PASS' if ok_g else 'FAIL'), f'| attr own {a[0]:.2f} ({a[1]}/{a[2]}) user {u[0]:.2f} ({u[1]}/{u[2]})',
      f"| game own {ga['own_share']} user {gu['own_share']} control {c['base_A']['accuracy']} {c['base_U']['accuracy']}")
EOF
)
st "3 smoke check: $verdict"
set -- $verdict
if [ "$1" = PASS ]; then
  for job in "all:$ALL:attr" "all_content:$CON:content" "all_arith:$CON:arith"; do
    stop
    name=${job%%:*}; rest=${job#*:}; conds=${rest%:*}; q=${rest##*:}
    touch thinking_swap/gpt_oss_120b_$name.GO
    st "4 attribution-side $name start"
    $PY crossmodel_role_swap.py $SW --question $q --conds $conds --out thinking_swap/gpt_oss_120b_$name > gptoss120b_$name.log 2>&1
    st "4 $name exit $?"
  done
fi
if [ "$2" = PASS ]; then
  for q in main control; do
    stop
    touch games_swap/gpt_oss_120b_s5_$q.GO
    st "4 game $q start"
    $PY $G --qkind $q --out games_swap/gpt_oss_120b_s5_$q > gptoss120b_g5_$q.log 2>&1
    st "4 game $q exit $?"
  done
fi
if [ "$1" = PASS ]; then
  stop
  O=thinking_swap/gpt_oss_120b_all_k64
  mkdir -p $O/items && cp -n thinking_swap/gpt_oss_120b_all/items/base_[AU]__*.json $O/items/ && touch $O.GO
  st "5 k64 start ($(ls $O/items | wc -l) unedited items copied)"
  $PY crossmodel_role_swap.py $SW --conds $K64 --out $O > gptoss120b_k64.log 2>&1
  st "5 k64 exit $?"
fi
st "done"
