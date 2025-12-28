# visualize_learning.py
"""
Interactive demonstration of how loss, backpropagation, and the
target network work together to train the AI.

Run this to SEE learning happen step-by-step!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("="*70)
print("🧠 VISUALIZING HOW NEURAL NETWORKS LEARN")
print("="*70)
print("\nThis shows you EXACTLY how loss and backpropagation work!")
print()

# ============================================================================
# PART 1: Simple Example - Learning to Predict a Number
# ============================================================================

print("PART 1: Simple Example - Learning to Predict")
print("-"*70)

class TinyNetwork(nn.Module):
    """Super simple network: 1 input → 1 output"""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0]))  # Start with weight=2
        
    def forward(self, x):
        return self.weight * x

print("Let's train a tiny network to predict: output = 5 × input")
print()

# Create network and optimizer
net = TinyNetwork()
optimizer = optim.SGD(net.parameters(), lr=0.1)

print(f"Initial weight: {net.weight.item():.2f}")
print()

# Training data
inputs = torch.tensor([1.0, 2.0, 3.0])
targets = torch.tensor([5.0, 10.0, 15.0])  # True function: 5 × input

print("Training for 10 steps:")
print()

for step in range(10):
    # Forward pass
    predictions = net(inputs)
    
    # Compute loss
    loss = ((predictions - targets) ** 2).mean()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Get gradient before update
    gradient = net.weight.grad.item()
    old_weight = net.weight.item()
    
    # Update weight
    optimizer.step()
    new_weight = net.weight.item()
    
    # Show what happened
    if step < 5 or step == 9:  # Show first 5 and last step
        print(f"Step {step + 1}:")
        print(f"  Weight: {old_weight:.3f} → {new_weight:.3f}")
        print(f"  Predictions: {predictions.detach().numpy()}")
        print(f"  Targets:     {targets.numpy()}")
        print(f"  Loss: {loss.item():.3f}")
        print(f"  Gradient: {gradient:.3f}")
        print()

print(f"Final weight: {net.weight.item():.3f} (should be close to 5.0!)")
print()

# ============================================================================
# PART 2: Understanding Loss Function
# ============================================================================

print("\n" + "="*70)
print("PART 2: Understanding Loss Function")
print("-"*70)

print("""
Loss = (Target - Prediction)²

Why square?
  Without squaring:
    Error 1: 10 - 8 = 2
    Error 2: 8 - 10 = -2
    Average = (2 + (-2)) / 2 = 0  ← Bad! Errors cancel!
  
  With squaring:
    Error 1: (10 - 8)² = 4
    Error 2: (8 - 10)² = 4
    Average = (4 + 4) / 2 = 4  ← Good! Both contribute!

Let's visualize different losses:
""")

predictions = [3, 5, 7, 9, 11]
target = 7

print(f"Target: {target}")
print()
for pred in predictions:
    loss = (target - pred) ** 2
    bar = "█" * int(loss)
    print(f"Pred: {pred:2d} → Loss: {loss:3d} {bar}")

print("\nSee? The further from target, the bigger the loss!")

# ============================================================================
# PART 3: Two Networks (Policy vs Target)
# ============================================================================

print("\n" + "="*70)
print("PART 3: Policy Network vs Target Network")
print("-"*70)

class SimpleQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)  # Simple layer
    
    def forward(self, x):
        return self.fc(x)

# Create both networks
policy_net = SimpleQNetwork()
target_net = SimpleQNetwork()

# Copy policy weights to target
target_net.load_state_dict(policy_net.state_dict())

print("Created two identical networks:")
print()

# Check they're the same
dummy_state = torch.randn(1, 4)
policy_output = policy_net(dummy_state)
target_output = target_net(dummy_state)

print(f"Policy network output: {policy_output.detach().numpy()[0]}")
print(f"Target network output: {target_output.detach().numpy()[0]}")
print("✓ They're identical!")
print()

# Now train ONLY policy network
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

print("Training ONLY policy network for 100 steps...")
for _ in range(100):
    dummy_state = torch.randn(1, 4)
    output = policy_net(dummy_state)
    loss = output.mean()  # Dummy loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Compare again
policy_output = policy_net(dummy_state)
target_output = target_net(dummy_state)

print()
print(f"Policy network output: {policy_output.detach().numpy()[0]}")
print(f"Target network output: {target_output.detach().numpy()[0]}")
print("✓ Now they're DIFFERENT!")
print()
print("Policy network learned, target network stayed frozen!")

# ============================================================================
# PART 4: Complete DQN Training Step
# ============================================================================

print("\n" + "="*70)
print("PART 4: Complete DQN Training Step")
print("-"*70)

print("""
Let's walk through ONE complete training step:

Scenario: Agent just moved LEFT in 2048
""")

# Simulate the data
reward = 2.0
gamma = 0.99

print(f"1. EXPERIENCE:")
print(f"   Action taken: LEFT")
print(f"   Reward received: {reward}")
print()

# Policy network forward pass (current state)
current_state = torch.randn(1, 4)
with torch.no_grad():
    current_q_values = policy_net(current_state)

print(f"2. POLICY NETWORK (current state):")
print(f"   Q(UP):    {current_q_values[0, 0].item():.3f}")
print(f"   Q(DOWN):  {current_q_values[0, 1].item():.3f}")
print(f"   Q(LEFT):  {current_q_values[0, 2].item():.3f}  ← We took this action")
print(f"   Q(RIGHT): {current_q_values[0, 3].item():.3f}")
print()

current_q_left = current_q_values[0, 2].item()

# Target network forward pass (next state)
next_state = torch.randn(1, 4)
with torch.no_grad():
    next_q_values = target_net(next_state)
    max_next_q = next_q_values.max().item()

print(f"3. TARGET NETWORK (next state - FROZEN):")
print(f"   Q(UP):    {next_q_values[0, 0].item():.3f}")
print(f"   Q(DOWN):  {next_q_values[0, 1].item():.3f}")
print(f"   Q(LEFT):  {next_q_values[0, 2].item():.3f}")
print(f"   Q(RIGHT): {next_q_values[0, 3].item():.3f}")
print(f"   Max Q:    {max_next_q:.3f}  ← Best future action")
print()

# Compute target
target_q_left = reward + gamma * max_next_q

print(f"4. COMPUTE TARGET (Bellman Equation):")
print(f"   Target = reward + γ × max(next Q)")
print(f"   Target = {reward} + {gamma} × {max_next_q:.3f}")
print(f"   Target = {target_q_left:.3f}")
print()

# Compute loss
loss_value = (target_q_left - current_q_left) ** 2

print(f"5. COMPUTE LOSS:")
print(f"   Current Q(LEFT): {current_q_left:.3f}")
print(f"   Target Q(LEFT):  {target_q_left:.3f}")
print(f"   Difference:      {target_q_left - current_q_left:.3f}")
print(f"   Loss (squared):  {loss_value:.3f}")
print()

if abs(target_q_left - current_q_left) > 0.5:
    print(f"   ⚠️  Big error! Need to adjust weights!")
else:
    print(f"   ✓ Small error, predictions are good!")

print()
print(f"6. BACKPROPAGATION:")
print(f"   (Calculates how to adjust each weight)")
print(f"   Working backwards through network...")
print(f"   Computing gradients...")
print()

print(f"7. UPDATE WEIGHTS:")
print(f"   For each weight:")
print(f"     new_weight = old_weight - learning_rate × gradient")
print(f"   ")
print(f"   Example:")
print(f"     old_weight = 0.532")
print(f"     gradient = -0.142")
print(f"     learning_rate = 0.001")
print(f"     new_weight = 0.532 - 0.001 × (-0.142)")
print(f"     new_weight = 0.532 + 0.000142")
print(f"     new_weight = 0.532142")
print()
print(f"   (This happens for ALL weights in the network!)")

# ============================================================================
# PART 5: Watching Loss Decrease Over Time
# ============================================================================

print("\n" + "="*70)
print("PART 5: Watching Loss Decrease During Training")
print("-"*70)

# Create fresh networks
policy_net = SimpleQNetwork()
target_net = SimpleQNetwork()
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

print("Training policy network for 50 steps...")
print("(Target network stays frozen)")
print()

losses = []
for step in range(50):
    # Generate fake experience
    state = torch.randn(1, 4)
    action_idx = 2  # LEFT
    reward = torch.tensor([2.0])
    next_state = torch.randn(1, 4)
    
    # Current Q
    current_q = policy_net(state)[0, action_idx]
    
    # Target Q (from frozen network)
    with torch.no_grad():
        max_next_q = target_net(next_state).max()
        target_q = reward + 0.99 * max_next_q
    
    # Loss
    loss = (current_q - target_q) ** 2
    losses.append(loss.item())
    
    # Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0 or step == 49:
        print(f"Step {step:2d}: Loss = {loss.item():.4f}")

print()
print("Loss trend:")
print(f"  Start: {losses[0]:.4f}")
print(f"  End:   {losses[-1]:.4f}")
print(f"  Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
print()
print("✓ Loss decreased! Network is learning!")

# ============================================================================
# PART 6: Why Update Target Network?
# ============================================================================

print("\n" + "="*70)
print("PART 6: When to Update Target Network")
print("-"*70)

print("""
Every 1000 steps, we copy policy → target:

Step 0-999:
  Policy network: Learning and changing
  Target network: Frozen (provides stable targets)
  
Step 1000:
  Copy: policy_net → target_net
  Now target has all the improvements!
  
Step 1000-1999:
  Policy network: Continues learning
  Target network: Frozen again (with new, better weights)
  
Step 2000:
  Copy again!

Why?
  ✓ Keeps targets stable (prevents "chasing tail")
  ✓ But not too stale (updates every 1000 steps)
  ✓ Best of both worlds!
""")

# ============================================================================
# SUMMARY
# ============================================================================

print("="*70)
print("📚 SUMMARY")
print("="*70)

print("""
🎯 LOSS FUNCTION:
   • Measures how wrong predictions are
   • Loss = (Target - Current)²
   • Bigger loss = worse predictions

🧠 TWO NETWORKS:
   • Policy Network: Being trained, updates every step
   • Target Network: Frozen, updates every 1000 steps
   • Prevents unstable learning

⚙️ BACKPROPAGATION:
   • Calculates how to change weights
   • Works backwards through network
   • Computes gradient for each weight

📈 GRADIENT DESCENT:
   • Updates weights using gradients
   • new_weight = old_weight - learning_rate × gradient
   • Tiny steps, millions of times → Smart AI!

🔄 THE LEARNING LOOP:
   1. Predict with policy network
   2. Compute target with target network
   3. Calculate loss
   4. Backpropagate
   5. Update policy weights
   6. Repeat millions of times!
   7. Every 1000 steps: copy policy → target

The magic: Network learns ENTIRELY from rewards!
No one tells it "move left here" - it figures it out! 🎉
""")

print("\n" + "="*70)
print("Now you understand how neural networks learn!")
print("="*70)