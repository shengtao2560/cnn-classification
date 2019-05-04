# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.drugs_loader import read_category, read_vocab


base_dir = 'data/drug'
vocab_dir = os.path.join(base_dir, 'drug.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)

    def predict(self, message):
        data = [self.word_to_id[x] for x in message if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    cnn_model = CnnModel()
    test_demo = ['药品降价死让患者无药可用，导致好心办坏事月日四合院时评近日，有媒体曝出，被称为乳腺癌救命药的赫赛汀自年纳入医保后在'
                 '中国多地出现缺货状态而在类进口药品取消关税的政策优惠下，这些进口药会否也出现降价就短缺的现象值得关注中国近期药品短'
                 '缺现象非偶发，在这些断缺药品中不乏廉价常用药，甚至包括救命药过去几年，药荒轮番上演年，治疗心脏衰竭的抢救用药西地兰'
                 '注射液短缺；年，治疗甲亢的他巴唑断货；年，心外科用药地高辛片放线菌素全国断供；年，儿童急性淋巴细胞白血病的必备药巯'
                 '嘌呤片断货此外，牛黄解毒丸红霉素消炎软膏氯霉素滴眼液等常用的廉价药都出现过断档或停产而许多药品都出现了降价就消失的'
                 '现象，让患者无所适从之所以会出现药品降价死，主要是因为这些药品缺乏足够的利润空间，让原料供应商药品生产企业与销售终'
                 '端间产生博弈，三者间的博弈却让患者承受了最终的苦果降价药廉价药品的短缺消失，让患者不得不选择价格高额的进口药而这一'
                 '结果显然与政府多次要求药品降价将药品纳入医保体系的初衷相违背的药品降价是为了让患者可降低药品花费，切实感受到廉价平'
                 '价药带来的好处，但药品降价死的现象却让患者面临无药可用的境况这种好心办坏事的情形一再发生，只会影响民众对政府政策的'
                 '信任和观感特别是随着进口药品关税降低等优惠政策的实施，未来药品市场势必将会发生重大改变患者能否在此次改变中获得实惠'
                 '，就需政府有关部门做好准备在与跨国药企价格谈判过程中，应当要求药企对中国市场的需求有所准备，优化药品生产与供应链，'
                 '保证相对稳定的供应量，避免出现进口降价药荒的现象其实，要满足患者用药的需求，除了依靠市场调节外，还需要政府有关部门'
                 '加强药品管理中国新近成立了国家医疗保障局，有望将过去分散于多个部门的医疗保障职能进行整合，相信能提供药品管理对政策'
                 '变化的因应能力而这有利于统一药品管理，为保证政策制定时信息畅通全面考虑市场供求提供有效方式',
                 '公司肝素原料药进入量价齐升通道提前备货充分享受涨价红利肝素价格方面公司标准肝素原料药上游肝素粗品受生猪出栏量减少及'
                 '环保压力影响价格激涨公司基于对肝素粗品未来价格走势判断提前建立低价格肝素粗品库存有效抵御成本上涨甚至受益于此波上涨'
                 '肝素量方面公司产品质量控制严格一直以来为等欧美优质客户供货近几年库存调整策略完成采购量逐年上涨同时公司今年开拓新客'
                 '户的依诺肝素钠制剂是低分子肝素制剂中销售规模最大的品种销售峰值一度达到亿美元随着仿制药的进入近几年原研销售随有萎缩'
                 '但年销售量仍可达到亿美元预计下游对原料药的需求最大公司一旦与达成长期合作关系公司出口量有望上新台阶未来几年公司标准'
                 '肝素原料药业务有望进入量价齐升通道国内延伸肝素产业链获批低分子肝素制剂专业学术营销叠加高质量替代放量潜力巨大年先后'
                 '批准公司的依诺肝素钠制剂那屈肝素钙制剂达肝素钠制剂年三种低分子肝素制剂销售万支销售收入万低分子肝素制剂产品预计国内'
                 '市场规模亿左右未分类低分子肝素制剂占据大约一半市场未来伴随药政审批严格公司预灌封分类产品有望逐渐完成市场取代公司低'
                 '分子肝素制剂目前基数尚低从样本医院数据来看销售收入万元同比增长成长潜力巨大随着产品生产规模提升毛利率也具备较大的提'
                 '升空间目前公司产品多省中标且初步建立家经销商家医院的经销网络并定期组织产品学术培训以专业化的学术营销带动产品的推广'
                 '销售公司低分子肝素制剂专业学术营销叠加高质量替代放量潜力巨大依诺肝素钠注射剂美国申报开启国际化未来针剂制剂出口潜在'
                 '龙头全球无菌针剂生产能力稀缺公司目前具备条认证的注射剂生产线条自建条收购同时通过收购健进获得了一条认证注射剂生产线'
                 '一支优秀的制剂研发申报团队和的海外销售渠道与合作开发年月公司向提交了依诺肝素钠注射剂的申报申请该产品目前在审核阶段'
                 '若顺利有望年底获批该产品市场潜力巨大目前原研和仿制药大约各占一半市场整体市场规模亿美金未来该产品一旦上市权益共享将'
                 '给公司带来巨大收益同时公司后续还有大批制剂处于中美双报研发阶段投资建议买入投资评级个月目标价元我们看好公司短中长期'
                 '逻辑中短期受益于原料药量价齐升长期受益于国内外制剂开拓预计公司年年的收入分别为亿净利润分别为亿元分别为元对应分别为'
                 '倍成长性突出首次给予买入的投资评级个月目标价为元相当于年倍的动态市盈率']
    for i in test_demo:
        print(cnn_model.predict(i))
