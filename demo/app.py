import os
import sys
import gradio as gr

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from demo.data_loader import (
    get_scene_list, load_interactive_data, load_precomputed_features,
    get_scene_images, get_scene_image_count
)


def _reconstruct_question(round_idx, tag, initial_description, previous_answers):
    import re
    # Build context: initial description + all previous answers
    context_parts = [initial_description] + list(previous_answers)
    context = " ".join(context_parts)

    # Extract key objects from the accumulated context
    nouns = re.findall(r'\b(a |the |there is |there are )([a-z]+(?:\s+[a-z]+){0,2})\b', context.lower())
    object_refs = [m[1] for m in nouns[:3]] if nouns else []
    object_str = ", ".join(object_refs) if object_refs else "objects"

    if round_idx == 0:
        question = (
            f"Could you continue describing and enriching the details about "
            f"{object_str} and their spatial relationships in the scene?"
        )
    else:
        if tag == 0:
            question = (
                f"Based on the previous descriptions, could you continue to ask for "
                f"detailed relationships between the objects mentioned and other objects "
                f"in the room? Focus on spatial connections among {object_str}."
            )
        else:
            question = (
                f"Can you try to describe different furniture and their spatial "
                f"relationships from your memory that haven't been covered yet? "
                f"Focus on objects other than {object_str}."
            )
    return question


def build_app():
    scene_list = get_scene_list()

    with gr.Blocks(title="IDeal Interactive Demo", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # IDeal Interactive 3D Scene Retrieval Demo
        **Interactive Cross-modal Learning for Text-3D Scene Retrieval** (NeurIPS 2025)
        """)

        with gr.Tabs():
            _build_auto_tab(app, scene_list)

        gr.Markdown("---\n**IDeal** — Interactive Cross-modal Learning for Text-3D Scene Retrieval | NeurIPS 2025")

    return app


def _build_auto_tab(block, scene_list):
    """Build the auto-replay tab."""
    with gr.Tab("🔄 Auto Replay"):
        gr.Markdown("""
        Results are read from `temp_results` data to show the IDeal interactive retrieval process.
        """)

        with gr.Row():
            scene_dropdown = gr.Dropdown(
                choices=scene_list, value=scene_list[0],
                label=f"Select Scene ({len(scene_list)} scenes)"
            )
            query_dropdown = gr.Dropdown(
                choices=[], value=None,
                label="Select Query (Query 1-10)", interactive=False
            )

        btn_replay = gr.Button("Run Retrieval Replay", variant="primary", size="lg")

        gr.Markdown("### Multi-round Interaction Details")
        gr.Markdown("Adaptive Questioning: Answer theta=2.7")
        chatbot = gr.Chatbot(label="Interactive Dialogue", height=400, type="tuples")

        gr.Markdown("### Fusion Retrieval Results (Top-5)")
        result_df = gr.Dataframe(
            headers=["Rank", "Scene ID", "Similarity Score"],
            label=" ", interactive=False
        )

        gr.Markdown("### Top-1 Scene Visualization")
        scene_gallery = gr.Gallery(
            label="Retrieved Scene Images",
            columns=5,
            height="400px",
            show_download_button=True
        )
        scene_info = gr.Markdown("")

        def _on_scene(scene_id):
            # Load queries for the selected scene
            records = load_interactive_data().get(scene_id, [])
            choices = [f"Query {i+1}" for i in range(len(records))]

            # Reset all display content when scene changes
            return [
                gr.update(
                    choices=choices,
                    value=choices[0] if choices else None,
                    interactive=len(choices) > 0
                ),  # query_dropdown
                [],  # chatbot
                [],  # result_df
                [],  # scene_gallery
                ""   # scene_info
            ]

        def _on_query(scene_id, qstr):
            # Reset display content when query changes
            return [
                [],  # chatbot
                [],  # result_df
                [],  # scene_gallery
                ""   # scene_info
            ]

        def _build_chat_messages(rec):
            """Build chatbot messages: user gives initial desc then answers, system asks questions."""
            description = rec.get("description", "")
            answers = rec.get("answers", [])
            tags = rec.get("tag", [])
            etps = rec.get("etps", [])

            messages = []
            for i in range(len(answers)):
                tag_val = tags[i] if i < len(tags) else 1
                question = _reconstruct_question(i, tag_val, description, answers[:i])
                answer = answers[i]
                msg_type = "detail probing" if tag_val == 0 else "exploration divergence"

                if i == 0:
                    # First turn: user provides initial description, then system asks
                    messages.append([None, f"💬 Initial query:\n{description}"])
                    messages.append([f"❓ Round {i+1} question ({msg_type}, entropy={etps[i]:.3f}):\n{question}",
                                     f"💬 Round {i+1} answer:\n{answer}"])
                else:
                    messages.append([f"❓ Round {i+1} question ({msg_type}, entropy={etps[i]:.3f}):\n{question}",
                                     f"💬 Round {i+1} answer:\n{answer}"])
            return messages

        def _replay(scene_id, qstr):
            records = load_interactive_data().get(scene_id, [])
            idx = int(qstr.replace("Query ", "")) - 1
            rec = records[idx]
            answers = rec.get("answers", [])
            tags = rec.get("tag", [])
            etps = rec.get("etps", [])
            description = rec.get("description", "")

            # Step 1: Stream chat messages one by one
            chat_messages = []
            yield [], [], [], ""  # Clear previous results

            # Show initial query
            chat_messages.append([None, f"💬 Initial query:\n{description}"])
            yield chat_messages, [], "", "Showing initial query..."

            import time
            time.sleep(1)  # Small delay for better UX

            # Show each round of Q&A
            for i in range(len(answers)):
                tag_val = tags[i] if i < len(tags) else 1
                question = _reconstruct_question(i, tag_val, description, answers[:i])
                answer = answers[i]
                msg_type = "detail probing" if tag_val == 0 else "exploration divergence"

                # Add question
                chat_messages.append([f"❓ Round {i+1} question ({msg_type}, entropy={etps[i]:.3f}):\n{question}", None])
                yield chat_messages, [], "", f"Round {i+1}: Asking question..."

                time.sleep(0.5)

                # Add answer
                chat_messages.append([None, f"💬 Round {i+1} answer:\n{answer}"])
                yield chat_messages, [], "", f"Round {i+1}: Received answer..."

                time.sleep(0.5)

            # Step 2: Compute retrieval after all messages are shown
            yield chat_messages, [], "", "⏳ Computing retrieval..."

            try:
                feat, tags_np, etps_np, sum_feat = load_precomputed_features()
                all_records = [r for recs in load_interactive_data().values() for r in recs]
                rec_idx = None
                for i, r in enumerate(all_records):
                    if r["scene_id"] == scene_id and r["description"] == rec["description"]:
                        rec_idx = i
                        break

                if rec_idx is not None:
                    from demo.retrieval_engine import get_engine
                    engine = get_engine()
                    interactive_feats = feat[rec_idx]
                    orig_emb = interactive_feats[0]
                    ans_embs = [interactive_feats[j] for j in range(1, min(3, len(interactive_feats)))]
                    results = engine.compute_fusion_result(orig_emb, ans_embs, k=5)
                    rows = [[r["rank"], r["scene_id"], f"{r['score']:.4f}"] for r in results]

                    # Get top-1 scene and load its images
                    top1_scene = results[0]["scene_id"]
                    top1_images = get_scene_images(top1_scene, max_images=10)
                    image_count = len(top1_images)

                    if image_count > 0:
                        info_text = f"**Retrieved Scene**: {top1_scene}\n\nShowing {image_count} RGB images from ScanNet"
                    else:
                        info_text = f"**Retrieved Scene**: {top1_scene}\n\n⚠️ No images available for this scene"
                else:
                    rows = [["-", "Not record found", "-"]]
                    results = []
                    info_text = "No retrieval results"
            except Exception as e:
                rows = [["-", f"Error: {str(e)[:60]}", "-"]]
                results = []
                info_text = f"Error: {str(e)}"

            # Step 3: Show final results
            # Load images for display
            final_images = []
            if results and len(results) > 0:
                top1_scene = results[0]["scene_id"]
                final_images = get_scene_images(top1_scene, max_images=10)

            yield chat_messages, rows, final_images, info_text

        scene_dropdown.change(fn=_on_scene, inputs=scene_dropdown, outputs=[query_dropdown, chatbot, result_df, scene_gallery, scene_info])
        query_dropdown.change(fn=_on_query, inputs=[scene_dropdown, query_dropdown], outputs=[chatbot, result_df, scene_gallery, scene_info])
        btn_replay.click(
            fn=_replay,
            inputs=[scene_dropdown, query_dropdown],
            outputs=[chatbot, result_df, scene_gallery, scene_info],
            queue=True  # Enable streaming
        )

        # Initialize
        init_records = load_interactive_data().get(scene_list[0], [])
        if init_records:
            init_choices = [f"Query {i+1}" for i in range(len(init_records))]
            scene_dropdown.value = scene_list[0]
            query_dropdown.choices = init_choices
            query_dropdown.value = init_choices[0]
            # Note: chatbot, result_df, scene_gallery, scene_info remain empty until button click


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IDeal Demo")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--share", action="store_true", help="Create public URL")
    args = parser.parse_args()

    print(f"Initializing IDeal demo on http://127.0.0.1:{args.port} ...")
    print("Loading model (this may take 10-30 seconds)...")
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)
