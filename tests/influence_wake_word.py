# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
#
# inference_pipeline = pipeline(
#     task=Tasks.auto_speech_recognition,
#     model='iic/speech_sanm_kws_phone-xiaoyun-commands-online', model_revision="master",
#     keywords="你好",
#     output_dir="./outputs/debug",
#     device='cpu',
#     chunk_size=[4, 8, 4],
#     encoder_chunk_look_back=0,
#     decoder_chunk_look_back=0,
# )
#
# rec_result = inference_pipeline(input='./kws_xiaoyunxiaoyun.wav')
# print(rec_result)

from funasr import AutoModel

def prepare_model(keywords):
    model = AutoModel(model="iic/speech_sanm_kws_phone-xiaoyun-commands-online",
                      keywords=keywords,
                      output_dir="./outputs/debug",
                      device='cpu',
                      chunk_size=[4, 8, 4],
                      encoder_chunk_look_back=0,
                      decoder_chunk_look_back=0,
                      disable_update=True
                     )

    res = model.generate(input='./kws_xiaoyunxiaoyun.wav')
    print(res)

if __name__ == "__main__":
    prepare_model("小云小云")
    prepare_model("为什么")