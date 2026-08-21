import pandas as pd
import numpy as np

OLD_FILE = "workspace_10000_legacy_model.csv"
NEW_FILE = "workspace_10000_sectioned.csv"

old = pd.read_csv(OLD_FILE)
new = pd.read_csv(NEW_FILE)

# Check that the same random actuator inputs were used
input_cols = [
    "ul_inner_mm",
    "ul_middle_mm",
    "ul_outer_mm",
    "uphi_inner_deg",
    "uphi_middle_deg",
    "uphi_outer_deg",
]

same_inputs = np.allclose(
    old[input_cols].to_numpy(),
    new[input_cols].to_numpy()
)

print("Same actuator samples:", same_inputs)

# Compare predicted tip positions
old_tip = old[
    ["tip_x_mm", "tip_y_mm", "tip_z_mm"]
].to_numpy()

new_tip = new[
    ["tip_x_mm", "tip_y_mm", "tip_z_mm"]
].to_numpy()

difference = np.linalg.norm(
    new_tip - old_tip,
    axis=1
)

print()
print("Workspace model comparison")
print("--------------------------")
print(f"Number of samples: {len(difference)}")
print(f"Mean tip difference: {difference.mean():.3f} mm")
print(f"Median tip difference: {np.median(difference):.3f} mm")
print(f"Maximum tip difference: {difference.max():.3f} mm")
print(
    f"Configurations changed: "
    f"{np.sum(difference > 1e-6)} / {len(difference)}"
)

# Save differences for later dissertation analysis
comparison = new[input_cols].copy()

comparison["legacy_tip_x_mm"] = old_tip[:, 0]
comparison["legacy_tip_y_mm"] = old_tip[:, 1]
comparison["legacy_tip_z_mm"] = old_tip[:, 2]

comparison["sectioned_tip_x_mm"] = new_tip[:, 0]
comparison["sectioned_tip_y_mm"] = new_tip[:, 1]
comparison["sectioned_tip_z_mm"] = new_tip[:, 2]

comparison["tip_difference_mm"] = difference

comparison.to_csv(
    "workspace_model_comparison.csv",
    index=False
)

print()
print("Saved: workspace_model_comparison.csv")