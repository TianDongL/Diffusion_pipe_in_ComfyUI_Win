from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("\n" + "=" * 60)
print("Diffusion_pipe_in_ComfyUI_Win Plugin Loaded")
print("=" * 60)
print(f"🎉Nodes loaded: {len(NODE_CLASS_MAPPINGS)}")
print("\nAvailable Nodes:")
for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"  • {display_name}")
print("=" * 60 + "\n")

 