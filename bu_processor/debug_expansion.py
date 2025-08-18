from bu_processor.query.heuristic_expander import HeuristicExpander

ex = HeuristicExpander()
query = 'insurance financial loss'
expansions = ex.expand(query, num=3)

print('Query:', repr(query))
print('Expansions:')
for i, exp in enumerate(expansions):
    print(f'  {i+1}: {repr(exp)}')
print()
print('Checking conditions:')
print('Has explain:', any('explain' in exp.lower() for exp in expansions))
print('Has coverage or policy:', any('coverage' in exp.lower() or 'policy' in exp.lower() for exp in expansions))

# Let's also debug the tokenization
import re
toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\-]+", query.lower())
print('Tokens:', toks)

_SYNONYMS = {
    "insurance": ["coverage", "policy"],
    "financial": ["monetary", "economic"],
    "loss": ["damage", "liability"],
}

for i, t in enumerate(toks):
    if t in _SYNONYMS:
        print(f'Token "{t}" has synonyms: {_SYNONYMS[t]}')
