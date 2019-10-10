from args import *
from model import *
from train import *
from data import *
from utils import *
from tensorboardX import SummaryWriter


### args
args = make_args()
print(args)
np.random.seed(123)
args.name = '{}_{}_{}_pre{}_drop{}_yield{}_{}'.format(
    args.model, args.layer_num, args.hidden_dim, args.feature_pre, args.dropout, args.yield_prob, args.comment)

### set up gpu
if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
else:
    print('Using CPU')
device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')

### use graph templates
template_name = 'graphs/template_small.dat'
train_name = 'graphs/train_small.dat'
if args.recompute_template or not os.path.isfile(template_name):
    graphs_train, nodes_par1s_train, nodes_par2s_train = load_graphs_lcg(data_dir='dataset/train_set/',
                                                                         stats_dir='dataset/')
    save_graph_list(graphs_train,train_name, has_par=True,nodes_par1_list=nodes_par1s_train,nodes_par2_list=nodes_par2s_train)
    print('Train graphs saved!', len(graphs_train))
    node_nums = [graph.number_of_nodes() for graph in graphs_train]
    edge_nums = [graph.number_of_edges() for graph in graphs_train]
    print('Num {}, Node {} {} {}, Edge {} {} {}'.format(
        len(graphs_train), min(node_nums), max(node_nums), sum(node_nums) / len(node_nums), min(edge_nums),
        max(edge_nums), sum(edge_nums) / len(edge_nums)))

    dataset_train = Dataset_sat(graphs_train, nodes_par1s_train, nodes_par2s_train, epoch_len=5000,
                                yield_prob=args.yield_prob, speedup=False)

    graph_templates, nodes_par1s, nodes_par2s = dataset_train.get_template()
    save_graph_list(graph_templates,template_name, has_par=True,nodes_par1_list=nodes_par1s,nodes_par2_list=nodes_par2s)
    print('Template saved!')
else:
    graph_templates, nodes_par1s, nodes_par2s = load_graph_list(template_name, has_par=True)
    print('Template loaded!')

# ##filter for small templates
# graph_templates_new = []
# nodes_par1s_new = []
# nodes_par2s_new = []
# for i in range(len(graph_templates)):
#     if graph_templates[i].number_of_nodes()<2000:
#         graph_templates_new.append(graph_templates[i])
#         nodes_par1s_new.append(nodes_par1s[i])
#         nodes_par2s_new.append(nodes_par2s[i])
# graph_templates = graph_templates_new
# nodes_par1s = nodes_par1s_new
# nodes_par2s = nodes_par2s_new

print('Template num', len(graph_templates))


# load stats
with open('dataset/' + 'lcg_stats.csv') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    stats = []
    for stat in data:
        stats.append(stat)
generator_list = []
for i in range(len(graph_templates)):
    # find clause num
    for stat in stats:
        if int(stat[1]) == len(nodes_par1s[i])//2:
            clause_num = int(int(stat[2])*args.clause_ratio)
            break
    for j in range(args.gen_graph_num):
        generator_list.append(graph_generator(
            graph_templates[i], len(nodes_par1s[i]), sample_size=args.sample_size, device=device, clause_num=clause_num))


input_dim = 3 # 3 node types
model = locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim, output_dim=args.output_dim,
                feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout).to(device)
model_fname = 'model/'+args.name+str(args.epoch_load)
model.load_state_dict(torch.load(model_fname, map_location=device))
model.to(device)
model.eval()
print('Model loaded!', model_fname)

test(args, generator_list, model, repeat=args.repeat)



