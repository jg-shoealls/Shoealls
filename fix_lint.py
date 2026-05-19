with open("src/validation/report.py", "r") as f:
    content = f.read()

content = content.replace("        if key_prefix == \"loss\": ax.set_yscale(\"log\")\n        else: ax.set_ylim(0, 1.05)", "        if key_prefix == \"loss\":\n            ax.set_yscale(\"log\")\n        else:\n            ax.set_ylim(0, 1.05)")

content = content.replace("    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.0, 2.0)", "    table.auto_set_font_size(False)\n    table.set_fontsize(10)\n    table.scale(1.0, 2.0)")

content = content.replace("        if row == 0: cell.set_facecolor(C_PRIMARY); cell.set_text_props(color=\"white\", fontproperties=_FONT_PROP)\n        else: cell.set_text_props(fontproperties=_FONT_PROP_LIGHT)", "        if row == 0:\n            cell.set_facecolor(C_PRIMARY)\n            cell.set_text_props(color=\"white\", fontproperties=_FONT_PROP)\n        else:\n            cell.set_text_props(fontproperties=_FONT_PROP_LIGHT)")

content = content.replace("    ax.set_xticks(x); ax.set_xticklabels(short_kr, fontproperties=_FONT_PROP_LIGHT, fontsize=8, rotation=30)", "    ax.set_xticks(x)\n    ax.set_xticklabels(short_kr, fontproperties=_FONT_PROP_LIGHT, fontsize=8, rotation=30)")

content = content.replace("        patch.set_facecolor(color); patch.set_alpha(0.5)", "        patch.set_facecolor(color)\n        patch.set_alpha(0.5)")

content = content.replace("    for label in ax.get_xticklabels(): label.set_fontproperties(_FONT_PROP_LIGHT); label.set_rotation(30)", "    for label in ax.get_xticklabels():\n        label.set_fontproperties(_FONT_PROP_LIGHT)\n        label.set_rotation(30)")

content = content.replace("    ax.set_yticks(range(len(names_kr))); ax.set_yticklabels(names_kr, fontproperties=_FONT_PROP_LIGHT)", "    ax.set_yticks(range(len(names_kr)))\n    ax.set_yticklabels(names_kr, fontproperties=_FONT_PROP_LIGHT)")

with open("src/validation/report.py", "w") as f:
    f.write(content)

print("fixed report lint")
