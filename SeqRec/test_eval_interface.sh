python test_eval_interface.py \
--state_dict_path "/mmu_vcg2_wjc_ssd/songzeyu/agent_workspace/LLM-SRec/SeqRec/sasrec/Movies_and_TV/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.dropout=0.5.pth" \
--dataset "Movies_and_TV" \
--data_dir "/mmu_vcg2_wjc_ssd/songzeyu/Amazon-Reviews-2023-trainval" \
--test_dir "/mmu_vcg2_wjc_ssd/songzeyu/Amazon-Reviews-2023-test" \
--device 0 \
--maxlen 128