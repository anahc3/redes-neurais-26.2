"""Exercise 1 - One step of backpropagation by hand, verified numerically.

Network: 2 inputs -> 2 hidden (tanh) -> 1 output (tanh), loss L = (y - yhat)**2.
Every number here is reported in the report by hand; this script exists only
to verify that arithmetic to >=4 decimals, so the derivation is defensible.
"""
import numpy as np

np.set_printoptions(precision=6, suppress=True)

x = np.array([0.5, -0.2])
y = 1.0

W1 = np.array([[0.3, -0.1], [0.2, 0.4]])
b1 = np.array([0.1, -0.2])
W2 = np.array([0.5, -0.3])
b2 = 0.2
eta = 0.3

# --- A: forward pass ---
z1 = W1 @ x + b1
h1 = np.tanh(z1)
u2 = W2 @ h1 + b2
yhat = np.tanh(u2)
L = (y - yhat) ** 2

print("--- Forward pass ---")
print(f"z1  = {z1}")
print(f"h1  = {h1}")
print(f"u2  = {u2:.6f}")
print(f"yhat = {yhat:.6f}")
print(f"L   = {L:.6f}")

# --- B: backward pass ---
dL_dyhat = -2.0 * (y - yhat)
dyhat_du2 = 1.0 - yhat ** 2
dL_du2 = dL_dyhat * dyhat_du2

dL_dW2 = dL_du2 * h1
dL_db2 = dL_du2

dL_dh1 = dL_du2 * W2
dh1_dz1 = 1.0 - h1 ** 2
dL_dz1 = dL_dh1 * dh1_dz1

dL_dW1 = np.outer(dL_dz1, x)
dL_db1 = dL_dz1

print("\n--- Backward pass ---")
print(f"dL/dyhat = {dL_dyhat:.6f}")
print(f"dyhat/du2 = {dyhat_du2:.6f}")
print(f"dL/du2 = {dL_du2:.6f}")
print(f"dL/dW2 = {dL_dW2}")
print(f"dL/db2 = {dL_db2:.6f}")
print(f"dL/dh1 = {dL_dh1}")
print(f"dh1/dz1 = {dh1_dz1}")
print(f"dL/dz1 = {dL_dz1}")
print(f"dL/dW1 =\n{dL_dW1}")
print(f"dL/db1 = {dL_db1}")

# --- C: parameter update ---
W1_new = W1 - eta * dL_dW1
b1_new = b1 - eta * dL_db1
W2_new = W2 - eta * dL_dW2
b2_new = b2 - eta * dL_db2

print("\n--- Updated parameters ---")
print(f"W1_new =\n{W1_new}")
print(f"b1_new = {b1_new}")
print(f"W2_new = {W2_new}")
print(f"b2_new = {b2_new:.6f}")

# verify: next forward pass with updated parameters
z1_new = W1_new @ x + b1_new
h1_new = np.tanh(z1_new)
u2_new = W2_new @ h1_new + b2_new
yhat_new = np.tanh(u2_new)
L_new = (y - yhat_new) ** 2

print("\n--- Next forward pass (sanity check) ---")
print(f"yhat_new = {yhat_new:.6f}")
print(f"L_new = {L_new:.6f} (was {L:.6f}; {'decreased' if L_new < L else 'increased'})")

if __name__ == "__main__":
    pass
