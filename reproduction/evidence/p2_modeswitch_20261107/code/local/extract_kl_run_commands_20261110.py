r"""Full command lines of every run that the 14 queue scripts of this archive launched (the KL runs, their render-only checks and
dry runs, the KL replacement runs, and the MiMo-V2.6 mechanism runs), for the paper-data package (user 11-10: the queue scripts are
not shipped; this file takes their place). Neither identity.json nor the run logs record argv, so every command is expanded from the
as-run queue script itself (sha256 checked), by that script's own template, with the literal values of its task calls and tables
read from the script text; nothing is executed. Downloads of weights are not runs and are not listed; the model revision a script
pins is given with the command. Each run command is then checked against the identity.json (fit_diag.json for the fits) of its
output folder in the local mirror: model, template or call kwargs, language, fixed date, gate, cpu offload, fit, order, band, region,
question, conditions, batch, max_new, items, games, qkind and the number of games of a smoke. The queue-log lines that name the
output (start, DONE, FAILED, exit) are attached as the record that it ran.
Output: results/kl_run_commands_20261110.json, a list of {queue, queue_sha256, step, out, local, cwd, unset, env, argv, command, log,
stdin, model_revision, log_lines, ran, identity_check}; `command` is the whole line (cd, environment, interpreter, script, arguments).
usage: extract_kl_run_commands_20261110.py [ARCHIVE]"""
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

A = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
CODE = A / 'code'
SCRIPTS = {
    'pro/modeswitch_pro_20261107.sh': 'c253072c8ac0cd4e27f5f94921983ebe0c920a08de0869dd2593829c902b353e',
    'pro/modeswitch_pro_20261107b.sh': '1ffe7a920cdcc9bc6ad28fa8f4faf097decc30fd6366e28aa0346a77054cb154',
    'pro/modeswitch_ctrl_pro_20261107.sh': '2f6f42dcb710b4965c4f45be5d8d9109ac5672c86fce55c08b9feeb30e5c6130',
    'pro/modeswitch_fn_pro_20261107.sh': 'd57b73af88947df240a62c1e929f1b7ca92064a846a66687eb9c0bac470f2ed5',
    'pro/modeswitch_fn_pro_20261107b.sh': 'b4b020ceb8d47b4e0e019d12983b0afbacfea807989bdcb71d6b814275907cee',
    'pro/modeswitch_pro_20261108.sh': 'a01d363ffd7ec48faa1684bad522d0e61970c745640350e1975966ffb0845944',
    'pro/modeswitch_nat10_pro_20261108.sh': '349b6a35491e53d661a64e42abe3352b414491a672466c1b9f8ab09099491445',
    'pro/modeswitch_moonlight_pro_20261108.sh': '5912b25b2ad8e266b53ba2978def9106dba378e704d938c36e19bac259aed9a1',
    'pro/modeswitch_main6_pro_20261108.sh': 'c8342e1a339f400ca8004906863f2f542ebf0409e6a984790ad6e8705738e46c',
    'pro/modeswitch_nat3_pro_20261108.sh': '492d70fed1447604a1261ba183afbd62e1c2871c3d07be53712fc3d85cc31724',
    'pro/modeswitch_swap_pro_20261108.sh': '4139f97c916295922ab436d79d2ecfa225fd58b10a50e30afaa0f1142701ee88',
    'pro/mech_mimo26_pro_20261108.sh': '0ab8317ed040e2d7d90ca25515c33ded93db4bc01dddf264998d2d46abb8aff0',
    'pro/mech_mimo26_ext_pro_20261108.sh': '86e08da5f7dea4146c1d1467c369fa973bf838f5a6b92e72e6ea15b0242c821a',
    'server3/modeswitch_glm53_s3_20261108.sh': '85fe9069f93a78a5233182328111608c9b17c44465c32aeabf0ca68b45219db1',
}
MIRROR = {'/root/p2_scale_20261015/crossmodel/': 'results/pro/', '/root/data/p2_small_checks_20261014/crossmodel/': 'results/server3/'}


def read(rel):
    b = (CODE / rel).read_bytes()
    assert hashlib.sha256(b).hexdigest() == SCRIPTS[rel], f'{rel} changed'
    return b.decode('utf-8')


def expand(s, V):
    for k in sorted(V, key=len, reverse=True):
        s = s.replace('${' + k + '}', V[k]).replace('$' + k, V[k])
    return s


def assigns(t):
    """top-level NAME=value assignments (several per line, ';'-separated), in order, values expanded; exported names and unset names"""
    V, exp, uns = {}, [], []
    lines = t.split('\n')
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^(\w+)='([^']*)$", ln)
        if m:  # multi-line single-quoted value (the model tables)
            buf = [m.group(2)]
            i += 1
            while not lines[i].endswith("'"):
                buf.append(lines[i])
                i += 1
            buf.append(lines[i][:-1])
            V[m.group(1)] = '\n'.join(buf)
            i += 1
            continue
        if re.match(r'^(export |unset |\w+=)', ln):
            lex = shlex.shlex(ln, posix=True, punctuation_chars=';')
            lex.whitespace_split = True
            seg, segs = [], []
            for tok in lex:
                if tok == ';':
                    segs.append(seg)
                    seg = []
                else:
                    seg.append(tok)
            segs.append(seg)
            for sg in segs:
                if not sg:
                    continue
                if sg[0] == 'unset':
                    uns += sg[1:]
                    continue
                ex = sg[0] == 'export'
                for tok in (sg[1:] if ex else sg):
                    if re.match(r'^\w+=', tok):
                        k, v = tok.split('=', 1)
                        V[k] = expand(v, V)
                        if ex:
                            exp.append(k)
                    else:
                        break
        i += 1
    return V, exp, uns


def task_calls(t, name='task'):
    out = []
    for ln in t.split('\n'):
        m = re.match(r'^\s*' + name + r'\s+(.*?)\s*(?:&&\s*\\)?$', ln)
        if m and not ln.lstrip().startswith(name + '()'):
            out.append(shlex.split(m.group(1)))
    return out


def rows(table):
    return [r.split('|') for r in table.split('\n') if r.strip()]


R = []


def add(queue, step, out, cwd, argv, env=(), unset=(), exports=None, log=None, stdin=None, rev=None, label=None):
    E = dict(exports or {})
    for k in unset:
        E.pop(k, None)
    env = list(env)
    prefix = (['env'] + [x for k in unset for x in ('-u', k)]) if unset else ['env']
    keys = [k for k in E if k not in dict(env)] + [k for k, _ in env]
    allv = {**E, **dict(env)}
    cmd = f'cd {cwd} && ' + ' '.join(prefix + [f'{k}={shlex.quote(allv[k])}' for k in keys] + [shlex.quote(a) for a in argv])
    loc = None
    for srv, l in MIRROR.items():
        if out and out.startswith(srv):
            loc = l + out[len(srv):]
    R.append(dict(queue='code/' + queue, queue_sha256=SCRIPTS[queue], step=step, out=out, local=loc, cwd=cwd, unset=list(unset),
                  env={k: allv[k] for k in keys}, argv=argv, command=cmd, log=log, stdin=stdin, model_revision=rev, label=label or out))


def lang_args(lang):
    return ['--lang', 'en', '--bodies-en', 'bodies_en_150.json'] if lang == 'en' else []


NET = ['HF_ENDPOINT', 'https://hf-mirror.com'], ['HF_HUB_DISABLE_XET', '1']


def main_1107(q):
    t = read(q)
    V, exp, _ = assigns(t)
    X = {k: V[k] for k in exp}
    for a in task_calls(t):
        a = [expand(x, V) for x in a]
        tag, m, kw, lang, gate, pp, net, mem, off, extra = a[0], V['H'] + '/' + a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9:]
        out = V['P'] + '/' + tag
        argv = [V['PY'], V['S'], '--model', m, '--tmpl-kwargs', kw, '--bodies', V['B1'], '--out', out, '--stop-file', V['STOP'],
                '--min-free-gib', '4', '--max-memory-gib', mem, '--cpu-offload-gib', off] + lang_args(lang) + (['--gate-g', gate] if gate else []) + extra
        env = ([tuple(x) for x in NET] if net == '1' else []) + [('CUDA_VISIBLE_DEVICES', '0'), ('PYTHONPATH', pp)]
        add(q, 'run', out, V['C'], argv, env, ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'] if net == '1' else [], X, out + '.log', label=tag)


def ctrl_1107(q):
    t = read(q)
    V, exp, _ = assigns(t)
    X = {k: V[k] for k in exp}
    for a in task_calls(t):
        a = [expand(x, V) for x in a]
        tag, m, kw, lang, pp, net, mem, extra = a[0], V['H'] + '/' + a[1], a[2], a[3], a[4], a[5], a[6], a[7:]
        out = V['P'] + '/' + tag
        argv = [V['PY'], V['S'], '--model', m, '--tmpl-kwargs', kw, '--bodies', V['B1'], '--out', out, '--stop-file', V['STOP'],
                '--max-memory-gib', mem] + lang_args(lang) + extra
        env = ([tuple(x) for x in NET] if net == '1' else []) + [('CUDA_VISIBLE_DEVICES', '0'), ('PYTHONPATH', pp)]
        add(q, 'run', out, V['C'], argv, env, ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'] if net == '1' else [], X, out + '.log', label='ctrl/' + tag)


def fn_1107(q):
    t = read(q)
    V, exp, uns = assigns(t)
    X = {k: V[k] for k in exp if k not in uns}
    for a in task_calls(t):
        tag, kw = a[0], expand(a[1], V)
        out = V['P'] + '/' + tag
        argv = [V['PY'], V['S'], '--model', V['M'], '--tmpl-kwargs', kw, '--bodies', V['B1'], '--out', out, '--stop-file', V['STOP'],
                '--max-memory-gib', '84', '--cpu-offload-gib', '52']
        add(q, 'run', out, V['C'], argv, [('CUDA_VISIBLE_DEVICES', '0'), ('PYTHONPATH', V['TF19'])], uns, X, out + '.log', label='ctrl/' + tag)


def main_1108(q):
    t = read(q)
    V, exp, _ = assigns(t)
    X = {k: V[k] for k in exp}
    for a in task_calls(t):
        a = [expand(x, V) for x in a]
        s, od, tag, m, lang, gate, pp, net, mem, extra = a[0], a[1], a[2], V['H'] + '/' + a[3], a[4], a[5], a[6], a[7], a[8], a[9:]
        out = od + '/' + tag
        argv = [V['PY'], s, '--model', m, '--tmpl-kwargs', '{}', '--bodies', V['B1'], '--out', out, '--stop-file', V['STOP'],
                '--max-memory-gib', mem] + lang_args(lang) + (['--gate-g', gate, '--min-free-gib', '4'] if gate else []) + extra
        env = ([tuple(x) for x in NET] if net == '1' else []) + [('CUDA_VISIBLE_DEVICES', '0'), ('PYTHONPATH', pp)]
        add(q, 'run', out, V['C'], argv, env, ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'] if net == '1' else [], X, out + '.log')


def nat10_main6(q, nat10):
    t = read(q)
    V, exp, _ = assigns(t)
    X = {k: V[k] for k in exp}
    OFF = [('HF_HUB_OFFLINE', '1'), ('TRANSFORMERS_OFFLINE', '1')]
    for r in rows(V['MODELS']):
        if nat10:
            tag, repo, rev, lang, kw, gb, wrap, mem, off = r
        else:
            tag, repo, rev, lang, kw, gb, mem = r
            wrap, off = '', '0'
        m = V['D'] + '/' + repo
        w = [wrap, m] if wrap else []
        e = lang_args(lang)
        add(q, 'render_check', V['M'] + '/' + tag + '.render.txt', V['C'],
            [V['PY']] + w + ['render_dump_20261108.py', '--model', m, '--tmpl-kwargs', kw, '--bodies', V['B1']] + e, OFF, (), X,
            V['M'] + '/' + tag + '.render.txt', rev=rev, label=tag)
        add(q, 'dry_run', V['M'] + '/dry_' + tag, V['C'],
            [V['PY']] + w + [V['F'], '--model', m, '--tmpl-kwargs', kw, '--bodies', V['B1']] + e + ['--out', V['M'] + '/dry_' + tag, '--dry-run'],
            OFF, (), X, V['M'] + '/' + tag + '.dry.log', rev=rev, label=tag)
        for s, od, extra in ((V['F'], V['P'], ['--min-free-gib', '4']), (V['X'], V['P'] + '/ctrl', [])):
            out = od + '/' + tag
            argv = [V['PY']] + ([wrap, m] if wrap else []) + [s, '--model', m, '--tmpl-kwargs', kw, '--bodies', V['B1'], '--out', out,
                                                              '--stop-file', V['STOP'], '--max-memory-gib', mem, '--cpu-offload-gib', off] + e + extra
            add(q, 'run', out, V['C'], argv, OFF + [('CUDA_VISIBLE_DEVICES', '0')], (), X, out + '.log', rev=rev)


def moonlight(q):
    t = read(q)
    V, exp, _ = assigns(t)
    X = {k: V[k] for k in exp}
    m, W, tag = V['m'], V['W'], V['tag']
    add(q, 'render_check', V['M'] + '/' + tag + '.render.txt', V['C'],
        [V['PY'], W, m, 'render_dump_20261108.py', '--model', m, '--tmpl-kwargs', '{}', '--bodies', V['B1']], (), (), X,
        V['M'] + '/' + tag + '.render.txt', label=tag)
    add(q, 'dry_run', V['M'] + '/dry_' + tag, V['C'],
        [V['PY'], W, m, 'mode_switch_20261107.py', '--model', m, '--tmpl-kwargs', '{}', '--bodies', V['B1'], '--out', V['M'] + '/dry_' + tag,
         '--dry-run'], (), (), X, V['M'] + '/' + tag + '.dry.log', label=tag)
    for s, out, extra in (('mode_switch_20261107.py', V['P'] + '/' + tag, ['--min-free-gib', '4']),
                          ('mode_switch_ctrl_20261107.py', V['P'] + '/ctrl/' + tag, [])):
        add(q, 'run', out, V['C'], [V['PY'], W, m, s, '--model', m, '--tmpl-kwargs', '{}', '--bodies', V['B1'], '--out', out, '--stop-file',
                                    V['STOP'], '--max-memory-gib', '44', '--cpu-offload-gib', '0'] + extra,
            [('CUDA_VISIBLE_DEVICES', '0')], (), X, out + '.log')


REVS = {}


def nat3(q):
    t = read(q)
    V, exp, _ = assigns(t)
    X = {k: V[k] for k in exp}
    for m in re.finditer(r'^dl (\S+) (\S+) (\S+) ', t, re.M):
        REVS[m.group(1)] = m.group(3)
    for a in task_calls(t):
        a = [expand(x, V) for x in a if x != '&&']
        order, od, tag, m, lang, kw, pp, net, mem, extra = a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9:]
        out = od + '/' + tag
        argv = [V['PY'], V['S'], '--order', order, '--model', m, '--tmpl-kwargs', kw, '--bodies', V['B1'], '--out', out, '--stop-file',
                V['STOP'], '--max-memory-gib', mem] + lang_args(lang) + extra
        if net == '1':
            E, U = [tuple(x) for x in NET], ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']
        else:
            E, U = [('HF_HUB_OFFLINE', '1'), ('TRANSFORMERS_OFFLINE', '1')], []
        rev = next((v for k, v in REVS.items() if tag.startswith(k)), None)
        add(q, 'dry_run', out, V['C'], argv + ['--dry-run'], E + [('PYTHONPATH', pp)], U, X, od + '/' + tag + '.dry.log', rev=rev, label=tag)
        add(q, 'run', out, V['C'], argv, E + [('CUDA_VISIBLE_DEVICES', '0'), ('PYTHONPATH', pp)], U, X, out + '.log', rev=rev)


def swap(q):
    t = read(q)
    V, exp, _ = assigns(t)
    X = {k: V[k] for k in exp}
    rev = re.search(r"revision='([0-9a-f]{40})'", t).group(1)
    for a in task_calls(t):
        a = [expand(x, V) for x in a if x != '&&']
        tag, m, lang, kw, fit, pp, net, mem, extra = a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8:]
        out = V['SW'] + '/' + tag
        argv = [V['PY'], V['S'], '--model', m, '--tmpl-kwargs', kw, '--bodies', V['B1'], '--fit', fit, '--out', out, '--stop-file', V['STOP'],
                '--max-memory-gib', mem] + lang_args(lang) + extra
        if net == '1':
            E, U = [tuple(x) for x in NET], ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']
        else:
            E, U = [('HF_HUB_OFFLINE', '1'), ('TRANSFORMERS_OFFLINE', '1')], []
        r = rev if tag == 'g120' else None
        add(q, 'dry_run', out, V['C'], argv + ['--dry-run'], E + [('PYTHONPATH', pp)], U, X, V['SW'] + '/' + tag + '.dry.log', rev=r, label=tag)
        add(q, 'run', out, V['C'], argv, E + [('CUDA_VISIBLE_DEVICES', '0'), ('PYTHONPATH', pp)], U, X, out + '.log', rev=r, label=tag)


def role_swap_argv(V, out, fit, region, conds, question, limit=None):
    return [V['PY'], V['S'], '--mode', 'thinking', '--model', V['MODEL'], '--on-kwargs', V['KW'], '--lang', V['LG'], '--bodies', V['BD'],
            '--splits', V['B'], '--fit-from', fit, '--band', V['BAND'], '--conds', conds, '--region', region, '--question', question,
            '--batch', V['BT'], '--max-new', '3072', '--max-memory-gib', V['G'], '--save-gen', '--out', out] + (['--limit-bodies', limit] if limit else [])


def mech(q):
    t = read(q)
    V, exp, _ = assigns(t)
    X = {k: V[k] for k in exp}
    C, FD, DIR, W = V['C'], V['FD'], V['DIR'], V['W']
    G0 = [('CUDA_VISIBLE_DEVICES', '0')]
    add(q, 'fits', C + '/' + FD, C, [V['PY'], V['S'], '--make-fits', FD, '--mode', 'thinking', '--model', V['MODEL'], '--on-kwargs', V['KW'],
                                     '--lang', V['LG'], '--bodies', V['BD'], '--splits', V['B'], '--band', V['BAND'], '--max-memory-gib', V['G']],
        G0, (), X, W + '/fits.log', label='fits')
    conds = re.findall(r'run_swap \$(?:SM|Q|O) \$FD/(\S+) (\S+) "([^"]+)" (\S+)( 2)?', t)
    sm = DIR + '_r05_smoke'
    names = [sm, DIR + '_r05q_all', DIR + '_content_all', DIR + '_arith_all']
    qn = ['attr', 'attr', 'content', 'arith']
    assert len(conds) == 3, conds
    specs = [conds[0], conds[1], conds[2], conds[2]]
    for nm, (fit, region, cd, question, lim), qq in zip(names, specs, qn):
        question = qq if question == '$qn' else question
        add(q, 'smoke' if nm == sm else 'run', C + '/' + nm, C, role_swap_argv(V, nm, FD + '/' + fit, region, cd, question, '2' if lim else None),
            G0, (), X, W + '/' + Path(nm).name + '.log', label=nm.split('/')[-1].replace('mimo_v26_', ''))
        if nm == sm:
            add(q, 'smoke_check', C + '/' + nm, C, [V['PY'], 'r05r06_smoke_check.py', nm, 'thinking'], (), (), X, W + '/status.log', label='smoke check')
    out = C + '/modeswitch/swap/m26'
    add(q, 'run', out, C, [V['PY'], 'mode_switch_swap_20261108.py', '--model', V['MODEL'], '--tmpl-kwargs', V['KW'], '--bodies', V['B'], '--fit',
                           FD + '/fit_m.pt', '--out', out, '--max-memory-gib', V['G'], '--stop-file', V['STOP']], G0, (), X, out + '.log',
        label='KL swap')


def mech_ext(q):
    t = read(q)
    V, exp, _ = assigns(t)
    X = {k: V[k] for k in exp}
    C, FD, DIR, W, Q = V['C'], V['FD'], V['DIR'], V['W'], V['Q']
    G0 = [('CUDA_VISIBLE_DEVICES', '0')]
    for spec in re.findall(r'"(\S+ fit_\w+ \w+ \S+)"', t):
        name, fit, reg, conds = spec.split(' ')
        O = DIR + '_' + name
        add(q, 'copy_unedited', C + '/' + O, C, ['sh', '-c', f'mkdir -p {O}/items && cp -n {Q}/items/base_* {O}/items/'], (), (), X, None,
            label=name + ' (unedited items of r05q_all)')
        add(q, 'run', C + '/' + O, C, role_swap_argv(V, O, FD + '/' + fit + '.pt', reg, 'base_A:assistant:-:-:0,base_U:user:-:-:0,' + conds, 'attr'),
            G0, (), X, W + '/' + Path(O).name + '.log', label=name)
    C8 = V['C8']
    GM = ['crossmodel_games.py', '--mode', 'thinking', '--model', V['MODEL'], '--on-kwargs', V['KW'], '--items', 'games/games_zh_secret5.json',
          '--games', 'secret', '--lang', V['LG'], '--fit-from', FD + '/fit_q.pt', '--band', V['BAND'], '--conds', C8, '--batch', V['BT'],
          '--max-new', '3072', '--max-memory-gib', V['G'], '--save-gen']
    for qk in ('main', 'control'):
        out = 'games_swap/mimo_v26_s5_smoke_' + qk
        add(q, 'smoke', C + '/' + out, C, [V['PY']] + GM + ['--qkind', qk, '--limit-bodies', '8', '--out', out], G0, (), X,
            W + '/game_smoke_' + qk + '.log', label='game smoke ' + qk)
    gate = re.search(r"<<'EOF'\n(.*?)\nEOF", t, re.S).group(1)
    add(q, 'smoke_check', C + '/games_swap', C, [V['PY'], '-'], (), (), X, W + '/status.log', stdin=gate, label='game smoke check')
    for qk in ('main', 'control'):
        out = 'games_swap/mimo_v26_s5_' + qk
        add(q, 'run_if_gate_passes', C + '/' + out, C, [V['PY']] + GM + ['--qkind', qk, '--out', out], G0, (), X, W + '/game_' + qk + '.log',
            label='game full ' + qk)


def glm53(q):
    t = read(q)
    V, exp, _ = assigns(t)
    X = {k: V[k] for k in exp}
    P, TAG, MD, KW = V['P'], V['TAG'], V['MD'], V['KW']
    add(q, 'render_check', P + '/' + TAG + '.render.txt', V['C'], [V['PY'], 'render_dump_20261108.py', '--model', MD, '--tmpl-kwargs', KW, '--bodies', V['B1']],
        (), (), X, P + '/' + TAG + '.render.txt', label=TAG)
    add(q, 'dry_run', P + '/dry_' + TAG, V['C'], [V['PY'], V['F'], '--model', MD, '--tmpl-kwargs', KW, '--bodies', V['B1'], '--out', P + '/dry_' + TAG,
                                                  '--dry-run'], (), (), X, P + '/' + TAG + '.dry.log', label=TAG)
    for s, od, extra in ((V['X'], P + '/ctrl', []), (V['F'], P, ['--min-free-gib', '4'])):
        out = od + '/' + TAG
        add(q, 'run', out, V['C'], [V['PY'], s, '--model', MD, '--tmpl-kwargs', KW, '--bodies', V['B1'], '--out', out, '--stop-file', V['STOP'],
                                    '--max-memory-gib', '40', '--cpu-offload-gib', '300'] + extra, (), (), X, out + '.log')


# ---------------------------------------------------------------------------------------------------------------------------------
def log_lines():
    """queue-log lines per (prefix, segment): prefix -> list of (time, message); a script started twice (v1, v2) is split at its marker"""
    L = []
    for f in (A / 'results/pro/modeswitch/queue.log', A / 'results/server3/modeswitch/queue.log'):
        for ln in f.read_text(encoding='utf-8', errors='replace').split('\n'):
            m = re.match(r'^\[(.+?)\] (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$', ln)
            if m:
                L.append((m.group(1), m.group(2), m.group(3)))
    return L


PREFIX = {'pro/modeswitch_pro_20261107.sh': 'modeswitch', 'pro/modeswitch_pro_20261107b.sh': 'modeswitch',
          'pro/modeswitch_ctrl_pro_20261107.sh': 'modeswitch-ctrl', 'pro/modeswitch_fn_pro_20261107.sh': 'modeswitch-fn',
          'pro/modeswitch_fn_pro_20261107b.sh': 'modeswitch-fn', 'pro/modeswitch_pro_20261108.sh': 'modeswitch-1108',
          'pro/modeswitch_nat10_pro_20261108.sh': 'nat10 RUN', 'pro/modeswitch_moonlight_pro_20261108.sh': 'nat10 moonlight',
          'pro/modeswitch_main6_pro_20261108.sh': 'main6 RUN', 'pro/modeswitch_nat3_pro_20261108.sh': 'nat3',
          'pro/modeswitch_swap_pro_20261108.sh': 'swap', 'pro/mech_mimo26_pro_20261108.sh': 'mech m26',
          'pro/mech_mimo26_ext_pro_20261108.sh': 'mech m26 ext', 'server3/modeswitch_glm53_s3_20261108.sh': 'glm53 ms'}
# scripts run as two versions under one log prefix: the message that ends the first version
SPLIT = {'modeswitch': 'stopped in qwen38_nt_switch (stop flag)', 'modeswitch-fn': 'v1 stopped;'}
FIRST = {'pro/modeswitch_pro_20261107.sh', 'pro/modeswitch_fn_pro_20261107.sh'}


def attach_logs(L):
    for r in R:
        pre = PREFIX[r['queue'][5:]]
        seg = [x for x in L if x[0] == pre]
        if pre in SPLIT:
            cut = next(i for i, x in enumerate(seg) if x[2].startswith(SPLIT[pre]))
            seg = seg[:cut + 1] if r['queue'][5:] in FIRST else seg[cut + 1:]
        tgt = r['label'] if r['label'] and not r['label'].startswith('/') else r['out']
        keys = {tgt, r['out'] or ''}
        if pre.startswith('mech'):
            keys = {r['label'].split(' (')[0]}
            if r['label'].startswith('game smoke'):
                keys = {'game smoke'}
            if r['label'].startswith('game full'):
                keys = {r['label']}
            if r['step'] == 'fits':
                keys = {'fits'}
            if r['step'] == 'smoke' and pre == 'mech m26':
                keys = {'smoke'}
            if r['label'] in ('content_all', 'arith_all'):
                keys = {r['label'].split('_')[0] + ' control'}
        hits = []
        for p, ts, msg in seg:
            for k in keys:
                if k and re.search(r'(?<![\w/.-])' + re.escape(k) + r'(?![\w.-])', msg):
                    hits.append(f'{ts} {msg[:160]}')
                    break
        r['log_lines'] = hits if r['step'] in ('run', 'fits', 'smoke', 'run_if_gate_passes') else []
        if r['step'] in ('run', 'fits', 'smoke', 'run_if_gate_passes'):
            r['ran'] = any(re.search(r'(^|\s)start\b|\bstart$|\bstart \(', h) for h in hits)
        else:  # render-only checks, dry runs, smoke checks, copies: the record they leave
            loc = r['local']
            rec = r['log'] if r['step'] in ('render_check', 'dry_run') else None
            for srv, l in MIRROR.items():
                if rec and rec.startswith(srv):
                    rec = l + rec[len(srv):]
            r['ran'] = bool(rec and (A / rec).exists()) if rec else None
            r['_rec'] = rec
    # a render or dry-run record written by two commands (Moonlight: the nat10 lane, whose check failed, then its own script) belongs
    # to the later one
    seen = {}
    for r in R:
        if r.get('_rec'):
            if r['_rec'] in seen:
                seen[r['_rec']]['ran'] = 'record overwritten by a later command (' + r['queue'] + ')'
            seen[r['_rec']] = r
    for r in R:
        r.pop('_rec', None)


def argval(argv, k, default=None):
    return argv[argv.index(k) + 1] if k in argv else default


def check_identity(r):
    if r['step'] not in ('run', 'fits', 'smoke') or not r['local']:
        return None
    d = A / r['local']
    f = d / ('fit_diag.json' if r['step'] == 'fits' else 'identity.json')
    if not f.exists():
        return 'no identity.json in the local mirror'
    I = json.loads(f.read_text(encoding='utf-8'))
    a = r['argv']
    bad = []

    def eq(name, got, want):
        if got != want:
            bad.append(f'{name}: identity {got!r} vs command {want!r}')
    s = next(x for x in a if x.endswith('.py') and ('mode_switch' in x or 'crossmodel' in x))  # not the loader wrapper
    model = argval(a, '--model')
    eq('model', I.get('model'), model)
    if 'mode_switch' in s:
        eq('tmpl_kwargs', I.get('tmpl_kwargs'), json.loads(argval(a, '--tmpl-kwargs')))
        eq('lang', I.get('lang'), argval(a, '--lang', 'zh'))
        eq('fixed_date', I.get('fixed_date'), argval(a, '--fixed-date'))
        if 'gate_g' in I:
            eq('gate_g', I.get('gate_g'), argval(a, '--gate-g'))
        if 'cpu_offload_gib' in I and '--cpu-offload-gib' in a:
            eq('cpu_offload_gib', float(I['cpu_offload_gib']), float(argval(a, '--cpu-offload-gib')))
        if 'fit' in I:
            eq('fit', I.get('fit'), argval(a, '--fit'))
        if 'order' in I:
            eq('order', I.get('order'), argval(a, '--order'))
    else:  # crossmodel_role_swap_r05.py, crossmodel_games.py
        eq('on_kwargs', I.get('on_kwargs'), argval(a, '--on-kwargs'))
        eq('lang', I.get('lang'), argval(a, '--lang'))
        lo, hi = argval(a, '--band').split('-')
        eq('band', I.get('band'), [int(lo), int(hi)])
        if r['step'] == 'fits':
            eq('bodies', I.get('bodies'), argval(a, '--bodies'))
        else:
            eq('fit_from', I.get('fit_from'), argval(a, '--fit-from'))
            eq('conditions', I.get('conditions'), [c.split(':')[0] for c in argval(a, '--conds').split(',')])
            eq('batch', I.get('batch'), int(argval(a, '--batch')))
            eq('max_new', I.get('max_new'), int(argval(a, '--max-new')))
            if 'games' in s:
                eq('items', I.get('items'), argval(a, '--items'))
                eq('games', I.get('games'), argval(a, '--games'))
                eq('qkind', I.get('qkind'), argval(a, '--qkind'))
                if '--limit-bodies' in a:
                    eq('n_items', I.get('n_items'), int(argval(a, '--limit-bodies')))
            else:
                eq('question', I.get('question'), argval(a, '--question'))
                eq('region', I.get('region'), argval(a, '--region'))
    return 'ok' if not bad else '; '.join(bad)


def main():
    main_1107('pro/modeswitch_pro_20261107.sh')
    main_1107('pro/modeswitch_pro_20261107b.sh')
    ctrl_1107('pro/modeswitch_ctrl_pro_20261107.sh')
    fn_1107('pro/modeswitch_fn_pro_20261107.sh')
    fn_1107('pro/modeswitch_fn_pro_20261107b.sh')
    main_1108('pro/modeswitch_pro_20261108.sh')
    nat10_main6('pro/modeswitch_nat10_pro_20261108.sh', True)
    moonlight('pro/modeswitch_moonlight_pro_20261108.sh')
    nat10_main6('pro/modeswitch_main6_pro_20261108.sh', False)
    nat3('pro/modeswitch_nat3_pro_20261108.sh')
    swap('pro/modeswitch_swap_pro_20261108.sh')
    mech('pro/mech_mimo26_pro_20261108.sh')
    mech_ext('pro/mech_mimo26_ext_pro_20261108.sh')
    glm53('server3/modeswitch_glm53_s3_20261108.sh')
    attach_logs(log_lines())
    for r in R:
        r['identity_check'] = check_identity(r) if r['ran'] else None
        del r['label']
    out = A / 'results' / 'kl_run_commands_20261110.json'
    out.write_text(json.dumps(R, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    from collections import Counter
    print(len(R), 'commands from', len({r['queue'] for r in R}), 'queue scripts;', dict(Counter(r['step'] for r in R)))
    print('ran:', dict(Counter((r['step'], r['ran']) for r in R)))
    print('identity:', dict(Counter(r['identity_check'] if r['identity_check'] in (None, 'ok', 'no identity.json in the local mirror') else 'MISMATCH'
                                   for r in R)))
    for r in R:
        if r['identity_check'] not in (None, 'ok'):
            print('  ', r['queue'], r['step'], r['local'], '|', r['identity_check'])
    for r in R:
        if r['step'] == 'run' and not r['ran']:
            print('  not started per queue log:', r['queue'], r['out'])
    print('written', out)


if __name__ == '__main__':
    main()
