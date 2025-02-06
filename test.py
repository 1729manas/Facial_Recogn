from matplotlib import pyplot as plt
import matplotlib.patches as patches


fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')


client_box = patches.FancyBboxPatch((0.5, 7.5), 2, 1, boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
ax.add_patch(client_box)
ax.text(1.5, 8, "Client\n(Mobile App)", fontsize=10, ha='center', va='center')


middleware_box = patches.FancyBboxPatch((3.5, 7.5), 2, 1, boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
ax.add_patch(middleware_box)
ax.text(4.5, 8, "Middleware\n(Auth, Validation)", fontsize=10, ha='center', va='center')


server_box = patches.FancyBboxPatch((2.5, 2), 5, 5, boxstyle="round,pad=0.3", edgecolor="black", facecolor="#f0f0f0")
ax.add_patch(server_box)
ax.text(5, 6.7, "Server Side", fontsize=10, ha='center', va='center', fontweight='bold')


api_endpoints_box = patches.FancyBboxPatch((4, 5.5), 2, 1, boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
ax.add_patch(api_endpoints_box)
ax.text(5, 6, "API Endpoints", fontsize=10, ha='center', va='center')


controllers_box = patches.FancyBboxPatch((4, 4), 2, 1, boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
ax.add_patch(controllers_box)
ax.text(5, 4.5, "Controllers", fontsize=10, ha='center', va='center')


blockchain_box = patches.FancyBboxPatch((6, 3), 2, 1, boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
ax.add_patch(blockchain_box)
ax.text(7, 3.5, "Blockchain\nNetwork", fontsize=10, ha='center', va='center')

db_box = patches.FancyBboxPatch((1, 3), 2, 1, boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
ax.add_patch(db_box)
ax.text(2, 3.5, "MySQL\nDatabase", fontsize=10, ha='center', va='center')

models_box = patches.FancyBboxPatch((2.5, 4.5), 2, 1, boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
ax.add_patch(models_box)
ax.text(3.5, 5, "Models", fontsize=10, ha='center', va='center')


arrowprops = dict(arrowstyle="->", color="black", lw=1.5)
ax.annotate("", xy=(3.5, 7.5), xytext=(2.5, 7.5), arrowprops=arrowprops)
ax.annotate("", xy=(5, 5.5), xytext=(5, 7.5), arrowprops=arrowprops) 
ax.annotate("", xy=(5, 4), xytext=(5, 5.5), arrowprops=arrowprops)
ax.annotate("", xy=(3.5, 4.5), xytext=(5, 4), arrowprops=arrowprops)  
ax.annotate("", xy=(2, 3), xytext=(3.5, 4.5), arrowprops=arrowprops) 
ax.annotate("", xy=(6, 3), xytext=(5, 4), arrowprops=arrowprops)  


plt.tight_layout()
plt.show()
