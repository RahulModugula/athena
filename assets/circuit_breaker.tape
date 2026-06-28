# VHS tape — renders the agent circuit-breaker demo to a GIF.
# Regenerate with:  vhs assets/circuit_breaker.tape
Output assets/circuit_breaker.gif

Set Shell "bash"
Set FontSize 18
Set Width 1180
Set Height 760
Set Padding 24
Set Theme "Catppuccin Mocha"

# Activate the project venv off-screen so the visible command is just `python …`.
Hide
Type "source .venv/bin/activate" Enter
Type "clear" Enter
Show

Type "python examples/agent_circuit_breaker.py"
Sleep 600ms
Enter

# First call loads the NLI model (~3s), then one step streams every ~0.5s.
Sleep 8s
# Linger on the circuit-breaker result.
Sleep 2s
