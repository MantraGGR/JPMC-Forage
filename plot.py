import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = pd.read_csv("Nat_Gas.csv")

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert Dates column to datetime
df["Dates"] = pd.to_datetime(df["Dates"])

# Plot
plt.figure(figsize=(6,4))
plt.plot(df["Dates"], df["Prices"], marker="o", linestyle="-")

plt.title("Prices over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()
