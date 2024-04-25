export OPENAI_API_KEY=...
SAVE_DIR=${1:-"test_results"}

# # gallery view
# python -m tasks.eval.show_gallery \
#     --root_dir ${SAVE_DIR} 

# # compare view
python -m tasks.eval.demo.show_compare \
    --root_dir ${SAVE_DIR} 

