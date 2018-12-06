Search.setIndex({docnames:["ikkuna","ikkuna.export","ikkuna.export.subscriber","ikkuna.models","ikkuna.utils","ikkuna.visualization","index","main","train","user_guide"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":1,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:55},filenames:["ikkuna.rst","ikkuna.export.rst","ikkuna.export.subscriber.rst","ikkuna.models.rst","ikkuna.utils.rst","ikkuna.visualization.rst","index.rst","main.rst","train.rst","user_guide.rst"],objects:{"":{main:[7,0,0,"-"],train:[8,0,0,"-"]},"ikkuna.export":{Exporter:[1,1,1,""],messages:[1,0,0,"-"],subscriber:[2,0,0,"-"]},"ikkuna.export.Exporter":{__call__:[1,2,1,""],_add_module_by_name:[1,2,1,""],_bias_cache:[1,3,1,""],_depth:[1,3,1,""],_epoch:[1,3,1,""],_freeze_module:[1,2,1,""],_global_step:[1,3,1,""],_is_training:[1,3,1,""],_model:[1,3,1,""],_module_filter:[1,3,1,""],_modules:[1,3,1,""],_train_step:[1,3,1,""],_weight_cache:[1,3,1,""],add_modules:[1,2,1,""],epoch_finished:[1,2,1,""],message_bus:[1,3,1,""],modules:[1,3,1,""],named_modules:[1,3,1,""],new_activations:[1,2,1,""],new_input_data:[1,2,1,""],new_layer_gradients:[1,2,1,""],new_loss:[1,2,1,""],new_output_and_labels:[1,2,1,""],new_parameter_gradients:[1,2,1,""],set_loss:[1,2,1,""],set_model:[1,2,1,""],step:[1,2,1,""],test:[1,2,1,""],train:[1,2,1,""]},"ikkuna.export.messages":{Message:[1,1,1,""],MessageBundle:[1,1,1,""],MessageBus:[1,1,1,""],ModuleMessage:[1,1,1,""],NetworkMessage:[1,1,1,""],get_default_bus:[1,4,1,""]},"ikkuna.export.messages.Message":{__init__:[1,2,1,""],data:[1,3,1,""],epoch:[1,3,1,""],global_step:[1,3,1,""],key:[1,3,1,""],kind:[1,3,1,""],tag:[1,3,1,""],train_step:[1,3,1,""]},"ikkuna.export.messages.MessageBundle":{__getattr__:[1,2,1,""],__init__:[1,2,1,""],_check_message:[1,2,1,""],add_message:[1,2,1,""],complete:[1,2,1,""],data:[1,3,1,""],epoch:[1,3,1,""],expected_kinds:[1,3,1,""],global_step:[1,3,1,""],key:[1,3,1,""],kinds:[1,3,1,""],train_step:[1,3,1,""]},"ikkuna.export.messages.MessageBus":{__init__:[1,2,1,""],deregister_data_topic:[1,2,1,""],deregister_meta_topic:[1,2,1,""],name:[1,3,1,""],publish_module_message:[1,2,1,""],publish_network_message:[1,2,1,""],register_data_topic:[1,2,1,""],register_meta_topic:[1,2,1,""],register_subscriber:[1,2,1,""]},"ikkuna.export.messages.ModuleMessage":{key:[1,3,1,""],module:[1,3,1,""]},"ikkuna.export.messages.NetworkMessage":{data:[1,3,1,""],key:[1,3,1,""]},"ikkuna.export.subscriber":{CallbackSubscriber:[2,1,1,""],ConditionNumberSubscriber:[2,1,1,""],HessianEigenSubscriber:[2,1,1,""],HistogramSubscriber:[2,1,1,""],MeanSubscriber:[2,1,1,""],MessageMeanSubscriber:[2,1,1,""],NormSubscriber:[2,1,1,""],PlotSubscriber:[2,1,1,""],RatioSubscriber:[2,1,1,""],SpectralNormSubscriber:[2,1,1,""],Subscriber:[2,1,1,""],Subscription:[2,1,1,""],SumSubscriber:[2,1,1,""],SynchronizedSubscription:[2,1,1,""],TestAccuracySubscriber:[2,1,1,""],TrainAccuracySubscriber:[2,1,1,""],VarianceSubscriber:[2,1,1,""]},"ikkuna.export.subscriber.CallbackSubscriber":{__init__:[2,2,1,""],compute:[2,2,1,""]},"ikkuna.export.subscriber.ConditionNumberSubscriber":{__init__:[2,2,1,""],compute:[2,2,1,""]},"ikkuna.export.subscriber.HessianEigenSubscriber":{__init__:[2,2,1,""],compute:[2,2,1,""]},"ikkuna.export.subscriber.HistogramSubscriber":{compute:[2,2,1,""]},"ikkuna.export.subscriber.MeanSubscriber":{compute:[2,2,1,""]},"ikkuna.export.subscriber.MessageMeanSubscriber":{compute:[2,2,1,""]},"ikkuna.export.subscriber.NormSubscriber":{compute:[2,2,1,""]},"ikkuna.export.subscriber.PlotSubscriber":{__init__:[2,2,1,""],_backend:[2,3,1,""],backend:[2,3,1,""],compute:[2,2,1,""]},"ikkuna.export.subscriber.RatioSubscriber":{__init__:[2,2,1,""],compute:[2,2,1,""]},"ikkuna.export.subscriber.SpectralNormSubscriber":{__init__:[2,2,1,""],compute:[2,2,1,""]},"ikkuna.export.subscriber.Subscriber":{__del__:[2,2,1,""],__init__:[2,2,1,""],_add_publication:[2,2,1,""],compute:[2,2,1,""],kinds:[2,3,1,""],message_bus:[2,3,1,""],process_messages:[2,2,1,""],publications:[2,3,1,""],receive_message:[2,2,1,""],subscriptions:[2,3,1,""]},"ikkuna.export.subscriber.Subscription":{__init__:[2,2,1,""],_handle_message:[2,2,1,""],_subsample:[2,3,1,""],_subscriber:[2,3,1,""],_tag:[2,3,1,""],counter:[2,3,1,""],handle_message:[2,2,1,""],kinds:[2,3,1,""]},"ikkuna.export.subscriber.SumSubscriber":{compute:[2,2,1,""]},"ikkuna.export.subscriber.SynchronizedSubscription":{_handle_message:[2,2,1,""],_new_round:[2,2,1,""]},"ikkuna.export.subscriber.TestAccuracySubscriber":{__init__:[2,2,1,""],_data_loader:[2,3,1,""],_dataset_meta:[2,3,1,""],_forward_fn:[2,3,1,""],_frequency:[2,3,1,""],compute:[2,2,1,""]},"ikkuna.export.subscriber.TrainAccuracySubscriber":{__init__:[2,2,1,""],compute:[2,2,1,""]},"ikkuna.export.subscriber.VarianceSubscriber":{compute:[2,2,1,""]},"ikkuna.models":{AlexNetMini:[3,1,1,""],DenseNet:[3,1,1,""],ResNet:[3,1,1,""],VGG:[3,1,1,""]},"ikkuna.models.AlexNetMini":{H_out:[3,3,1,""],W_out:[3,3,1,""],__init__:[3,2,1,""],classifier:[3,3,1,""],features:[3,3,1,""],forward:[3,2,1,""]},"ikkuna.models.DenseNet":{forward:[3,2,1,""]},"ikkuna.models.ResNet":{forward:[3,2,1,""]},"ikkuna.models.VGG":{forward:[3,2,1,""]},"ikkuna.utils":{ModuleTree:[4,1,1,""],NamedModule:[4,1,1,""],available_optimizers:[4,4,1,""],create_optimizer:[4,4,1,""],initialize_model:[4,4,1,""],load_dataset:[4,4,1,""],make_fill_polygons:[4,4,1,""],numba:[4,0,0,"-"]},"ikkuna.utils.ModuleTree":{__init__:[4,2,1,""],_children:[4,3,1,""],_module:[4,3,1,""],_name:[4,3,1,""],_type_counter:[4,3,1,""],preorder:[4,2,1,""]},"ikkuna.utils.NamedModule":{__getnewargs__:[4,2,1,""],__new__:[4,5,1,""],__repr__:[4,2,1,""],_asdict:[4,2,1,""],_make:[4,6,1,""],_replace:[4,2,1,""],module:[4,3,1,""],name:[4,3,1,""]},"ikkuna.utils.numba":{compute_bin:[4,3,1,""],dtype_min_max:[4,4,1,""],numba_gpu_histogram:[4,4,1,""],tensor_to_numba:[4,4,1,""],typestr:[4,4,1,""]},"ikkuna.visualization":{Backend:[5,1,1,""],MPLBackend:[5,1,1,""],TBBackend:[5,1,1,""],configure_prefix:[5,4,1,""]},"ikkuna.visualization.Backend":{__init__:[5,2,1,""],add_data:[5,2,1,""],add_histogram:[5,2,1,""],title:[5,3,1,""]},"ikkuna.visualization.MPLBackend":{__init__:[5,2,1,""],_axes:[5,3,1,""],_buffer:[5,3,1,""],_buffer_lim:[5,3,1,""],_plots:[5,3,1,""],_prepare_axis:[5,2,1,""],_redraw_counter:[5,3,1,""],_reflow_plots:[5,2,1,""],_xlabel:[5,3,1,""],_ylabel:[5,3,1,""],_ylims:[5,3,1,""],add_data:[5,2,1,""],add_histogram:[5,2,1,""],title:[5,3,1,""],xlabel:[5,3,1,""],ylabel:[5,3,1,""]},"ikkuna.visualization.TBBackend":{_hist_bins:[5,3,1,""],_writer:[5,3,1,""],add_data:[5,2,1,""],add_histogram:[5,2,1,""],info:[5,3,1,""]},"train.DatasetMeta":{__getnewargs__:[8,2,1,""],__new__:[8,5,1,""],__repr__:[8,2,1,""],_asdict:[8,2,1,""],_make:[8,6,1,""],_replace:[8,2,1,""],dataset:[8,3,1,""],num_classes:[8,3,1,""],shape:[8,3,1,""],size:[8,3,1,""]},"train.Trainer":{__init__:[8,2,1,""],_batch_size:[8,3,1,""],_dataloader:[8,3,1,""],_dataset:[8,3,1,""],_loss_function:[8,3,1,""],_num_classes:[8,3,1,""],_optimizer:[8,3,1,""],_scheduler:[8,3,1,""],_shape:[8,3,1,""],add_subscriber:[8,2,1,""],batches_per_epoch:[8,3,1,""],create_graph:[8,3,1,""],current_batch:[8,3,1,""],exporter:[8,3,1,""],initialize:[8,2,1,""],loss:[8,3,1,""],model:[8,3,1,""],optimize:[8,2,1,""],optimizer:[8,3,1,""],set_model:[8,2,1,""],set_schedule:[8,2,1,""],train_batch:[8,2,1,""]},ikkuna:{"export":[1,0,0,"-"],models:[3,0,0,"-"],utils:[4,0,0,"-"],visualization:[5,0,0,"-"]},main:{_main:[7,4,1,""],get_parser:[7,4,1,""],main:[7,4,1,""]},train:{DatasetMeta:[8,1,1,""],Trainer:[8,1,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"],"5":["py","staticmethod","Python static method"],"6":["py","classmethod","Python class method"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function","5":"py:staticmethod","6":"py:classmethod"},terms:{"32x32":3,"case":[1,9],"class":[1,2,3,4,5,7,8,9],"default":[1,4,7,8],"export":[0,3,6,8,9],"final":[1,9],"float":[1,3,4],"function":[1,2,3,4,6,7,8,9],"import":9,"int":[1,2,3,4,5,7,8,9],"new":[1,2,4,6,8],"public":2,"return":[1,4,7,8,9],"static":[4,8],"super":9,"switch":1,"true":[1,2,3,4,9],"while":3,Adding:1,For:[2,5,9],Not:2,The:[1,2,4,5,6,8,9],Their:9,There:9,These:[1,2,9],Use:7,Used:[4,8],Useful:4,Using:9,Will:[1,4],__call__:1,__cuda_array_interface__:4,__del__:2,__getattr__:1,__getnewargs__:[4,8],__init__:[1,2,3,4,5,8,9],__new__:[4,8],__repr__:[4,8],_add_module_by_nam:1,_add_publ:[2,9],_asdict:[4,8],_ax:5,_backend:[2,9],_batch_siz:8,_bias_cach:1,_buffer:5,_buffer_lim:5,_check_messag:1,_children:4,_cl:[4,8],_condition_numb:2,_data_kind:9,_data_load:2,_dataload:8,_dataset:8,_dataset_meta:2,_depth:1,_epoch:1,_forward_fn:2,_freeze_modul:1,_frequenc:2,_global_step:1,_handle_messag:2,_hist_bin:5,_is_train:1,_loss:[1,8],_loss_funct:8,_lrschedul:8,_main:7,_make:[4,8],_mean:2,_message_mean:2,_meta_kind:9,_metric_postprocess:9,_model:1,_modul:[1,4],_module_filt:1,_name:4,_new_round:2,_norm:2,_num_class:8,_optim:8,_plot:5,_prepare_axi:5,_ratio:[2,9],_redraw_count:5,_reflow_plot:5,_replac:[4,8],_run:5,_schedul:8,_shape:8,_spectral_norm:2,_subsampl:2,_subscrib:2,_sum:2,_tag:2,_train_step:1,_type_count:4,_varianc:2,_very_:2,_weight_cach:1,_writer:5,_xlabel:5,_ylabel:5,_ylim:5,a_gpu:4,abc:[1,5],abl:9,about:[8,9],abs:9,absolut:[2,9],abund:5,accept:[2,7],access:[1,8,9],accross:1,accuraci:[2,7],across:1,activ:[1,2],actual:2,adam:[7,8],adapt:[3,9],add:[1,3,4,7,8,9],add_data:[5,9],add_histogram:5,add_messag:1,add_modul:[1,9],add_subscrib:8,adding:9,after:[1,2,3,9],afterward:3,ai_ikkuna:9,albeit:9,alexnet:[3,4,9],alexnetmini:[3,4,9],alia:[1,4,8],all:[1,2,3,4,6,8,9],allow:9,along:9,alreadi:1,also:[2,9],altern:9,although:3,alwai:[1,2,9],ani:[1,2,9],ann:7,anneal:8,announc:9,anyth:9,appear:4,append:[5,9],appli:2,appropri:1,arbitrari:[3,9],arg:[1,8,9],argpars:7,argument:[4,6,9],argumentpars:7,around:[2,8],arrai:4,arrang:5,arriv:[1,2],art3d:4,artifact:[1,2],arxiv:3,aspect:1,assembl:1,associ:[1,2],assum:[3,4,8,9],attach:[1,2,9],attempt:[3,8,9],attr:9,attribut:[4,9],author:9,author_email:9,auto:5,automat:8,avail:[4,8,9],available_optim:4,averag:[2,7,9],axi:[2,5],backend:[2,5,7,9],bare:5,base:[1,2,3,4,5,8],basic:[3,9],batch:[1,2,7,8],batch_finish:2,batch_siz:[2,7,8],batches_per_epoch:8,becom:5,been:[1,2,9],befor:[2,5,8,9],begin:8,behind:9,being:9,belong:1,below:[1,4],besid:1,better:2,between:[2,7,9],bewar:3,bia:[1,4],bias:4,bias_val:4,bin:[4,5,9],bleed:9,block:3,block_config:3,blocktyp:3,bn_size:3,bool:[1,2,3,4,9],bottl:3,bottleneck:3,bound:2,box:9,buffer:[1,2,5],buffer_lim:5,bug:9,built:[4,8],bundl:[1,2,8,9],bus:[1,9],cach:[1,2],call:[1,2,3,8,9],callback:[1,2],callbacksubscrib:2,can:[1,2,4,9],cannot:2,captur:1,care:[3,9],carri:1,categori:8,ceas:2,central:9,certain:9,chang:1,channel:[3,9],check:1,checkpoint:3,child:4,children:4,choic:7,chosen:9,cifar100:7,cifar10:7,cifar:3,classif:3,classifi:[3,7,9],classmethod:[4,8],clear:[1,2],code:[1,9],collect:1,com:9,come:9,common:1,compil:9,complet:[1,2],compress:3,comput:[1,2,3,4,9],compute_bin:4,conda:9,condit:2,conditionnumbersubscrib:2,configur:[2,7],configure_prefix:5,connect:[3,9],consid:9,consist:1,constant:4,construct:1,constructor:[8,9],consum:[5,9],contain:[1,4,7,9],content:6,conv2d:[1,9],conv:[3,4,9],conveni:[1,9],converg:2,convert:4,convnet:9,convolut:[3,9],copi:[4,8],core:9,correct:[2,9],could:2,counter:[1,2,9],creat:[1,4,6,8],create_graph:8,create_optim:4,crossentropyloss:[2,8],cuda:[4,8],current:[1,2,4,7,8,9],current_batch:8,cutoff:[3,9],data:[1,2,4,5,8,9],data_load:2,dataload:[2,8],datapoint:5,dataset:[2,4,7,8],dataset_meta:[2,8],dataset_str:7,datasetmeta:[2,4,8],datum:5,decai:7,decomposit:2,def:9,defalt:2,defin:[3,9],delet:2,deliv:2,denot:1,dens:3,densenet:[3,4],depend:2,depth:[1,4,7,8],deregister_data_top:1,deregister_meta_top:1,descend:1,descript:9,desir:[2,9],detail:6,determin:[2,4],develop:9,devic:4,dict:[1,2,4,5,8],dictionari:9,differ:[1,2,3,4],differenti:2,dim:9,dir:7,direct:1,directori:[4,5],disabl:1,disambigu:4,dispatch:5,displai:[2,5,9],disregard:4,distutil:9,divid:8,dividend:[2,9],divisor:[2,9],document:[4,9],doe:[1,2,5,8],doesn:2,don:[7,9],done:2,drop:[4,8],drop_nam:4,drop_rat:3,dropout:[3,9],dtype:4,dtype_min_max:4,due:5,dunno:1,duplic:1,dure:[1,8],each:[1,2,3,4,9],easi:9,easier:6,easili:8,edg:9,effici:3,egg:9,eigenvalu:2,either:[1,2,9],element:[2,9],els:[2,9],email:9,emit:[1,5],empti:4,enabl:[1,2],encapsul:8,encount:9,end:2,enforc:1,enough:[3,9],ensur:1,entir:[2,9],entry_point:9,env:9,environ:9,epoch:[1,2,7,8,9],epoch_finish:[1,2,9],error:9,estim:[2,4],evenli:8,event:9,everi:[2,3,4,9],everyth:[1,9],exact:2,exampl:[8,9],exist:[2,8],expect:[1,9],expected_kind:1,expens:2,experi:2,explicitli:1,exponenti:7,extens:[4,7],extract:[1,3,9],extremely_complex_model:1,fact:5,factor:[2,3],fail:1,fals:[3,4,7],fashionmnist:7,featur:[1,3,9],few:[3,9],field:[1,4,8],figur:5,file:[4,5,9],fill:4,filter:[2,3,9],first:[2,3,9],fit:2,fix:2,flag:1,folder:4,follow:[1,7,9],form:[4,9],format:[4,8],former:[1,3],forward:[2,3,5,8,9],forward_fn:2,framework:9,freez:1,frequenc:2,from:[1,2,3,4,8,9],full:[2,8],funnction:8,further:[1,9],futur:4,gamma:7,gener:[2,4],get:[1,2,4,9],get_default_bu:[1,9],get_pars:7,git:9,github:9,given:[2,4,9],global:[1,2,5],global_step:[1,9],golmant:2,got:9,gradient:[1,2,9],grid:5,group:1,growth_rat:3,h_out:[3,9],hacki:4,had:9,handle_messag:2,handler:8,happen:[2,9],has:[1,2,9],have:[1,2,5,9],height:[3,9],here:2,hessian:[2,7],hessian_eigenth:2,hessianeigensubscrib:2,hierarch:4,high:5,histogram:[2,4,5,7],histogramsubscrib:2,hold:1,home:4,homogen:1,hook:[1,3,9],how:[1,2,3],http:[3,9],identifi:[1,2,9],ignor:[1,2,3,4,7,8],ikkuna:[8,9],imag:[3,4],imagenetdog:4,implement:2,in_:1,incom:2,increas:[1,2,7],index:[1,6,8],individu:9,infer:8,infinit:4,info:5,inform:[1,8,9],init:8,initi:[4,8,9],initialis:9,initialize_model:4,initil:8,inlin:5,inplac:9,input:[1,3,8,9],input_label:9,input_shap:[3,9],insid:9,insight:2,instal:[2,6],instanc:[3,4,8,9],instead:[1,2,3],interest:[1,2,9],intern:9,interoper:4,interpret:2,introduct:6,invers:2,investig:2,invok:9,involv:2,isn:4,issu:6,item:9,iter:[1,2,4,8],its:1,itself:5,jpg:4,jupyt:5,just:[2,3,9],keep:[1,4],kei:[1,2,9],kernel_s:9,keyword:[4,8],kind1:[2,9],kind2:[2,9],kind:[1,2,9],known:4,kwarg:[4,5,7,8],kwd:[4,8],label:[1,2,4,5,9],lag:9,lambda:9,lambdalr:8,larg:[3,9],larger:3,last:8,latter:[1,3],layer:[1,2,3,4,9],lead:2,learn:[3,7,8],learning_r:7,least:9,left:8,len:[4,8,9],less:6,lest:4,level:1,librari:[4,6],like:9,limit:[2,5],line:[2,4,5,9],linear:[1,2,9],list:[1,2,3,4,8,9],live:6,load:[1,8],load_dataset:[4,8],loader:[2,8],local:[1,9],log:[5,7],log_dir:7,logdir:7,logic:8,longer:1,look:9,loss:[1,2,8,9],loss_fn:2,loss_funct:1,lr_schedul:8,magic:2,mai:[4,9],main:6,make:[2,4,6,8],make_fill_polygon:4,manag:1,mandatori:1,mani:[2,3],map:[4,5,8],match:1,matplotlib:5,matric:2,matrix:2,max:[3,4,9],maxpool2d:9,mayb:[2,9],mean:[1,2,4,7,9],meansubscrib:2,member:1,memori:[2,3],mere:2,messag:[0,2,6,9],message_bu:[1,2,9],message_bundl:[1,2,9],message_or_bundl:2,messagebu:[1,2,9],messagebundl:[1,2,9],messagemeansubscrib:2,messsag:1,meta:[1,2],method:[1,2,4,8,9],metric:[1,2,5,9],mimick:1,min:4,mnist:7,mode:1,model:[0,1,2,4,6,7,8,9],model_or_str:8,model_str:7,modul:[0,6,7,9],module_filt:1,module_nam:[5,9],modulemessag:[1,2,9],moduletre:4,monitor:3,more:[2,3,5,6,8,9],most:9,mpl:7,mpl_toolkit:4,mplbackend:5,mplot3d:4,msg:1,much:3,multipl:[1,2,3],must:[2,8,9],mutlipl:1,name:[1,2,4,5,6,8,9],named_children:4,named_modul:1,namedmodul:[1,2,4],natur:2,necessari:[1,7],neck:3,need:[2,3,9],neither:1,network:[1,3,9],network_output:[1,9],networkmessag:[1,2,9],new_activ:1,new_input_data:1,new_layer_gradi:1,new_loss:1,new_output_and_label:1,new_parameter_gradi:1,newli:[1,2,9],nice:[4,8],noah:2,non:2,none:[1,2,3,4,7,9],nor:1,norm:[2,7,9],normal:2,normsubscrib:2,note:2,now:[1,5,9],nth:2,num_block:3,num_class:[3,8,9],num_eig:2,num_init_featur:3,numba:[0,6],numba_gpu_histogram:4,number:[1,2,3,4,5,7,8],numer:4,object:[1,2,4,8,9],obtain:[2,4,7,8],off:1,often:[2,5],older:9,onc:[1,2,9],one:[1,2,3,4,9],onli:[1,2,8,9],oper:2,optim:[4,7,8],option:[1,2,3,9],order:[2,3,9],ordereddict:[4,8],org:3,other:[1,2,6,8,9],otherwis:[1,2,3],out:[1,9],out_:1,output:[1,2,3,9],over:[2,5],overrid:[1,2,9],overridden:3,overwrit:9,own:9,packag:[2,6,9],pad:9,page:6,paper:3,param:[1,4],paramet:[1,2,3,4,5,7,8,9],parent:4,parser:7,part:5,pass:[2,3,4,8,9],payload:[1,5],pdf:3,peltarion:9,per:[1,2,5],perform:[3,4],person:6,perus:9,pickl:[4,8],pip:9,place:2,plain:[4,8],pleas:9,plot:[2,5,9],plot_config:[2,9],plotsubscrib:[2,9],png:4,point:1,poly3dcollect:4,polygon:4,pool:[3,9],posit:4,possibl:[1,2,7],power:2,power_iter_step:2,power_step:2,practic:2,predict:2,prefix:5,preorder:4,prepar:5,prerequisit:6,presenc:4,present:[4,9],previou:[1,2],previous:1,print:7,probabl:[2,9],procedur:7,process:[1,2,9],process_messag:2,program:6,progress:7,project:9,proper:2,properti:[1,2,9],provid:[3,9],publish:[1,2,9],publish_module_messag:[1,9],publish_network_messag:1,punctuat:5,push:2,pypi:9,python:9,pytorch:[2,9],quantiti:[2,9],question:1,quickstart:6,rais:[1,2,4,9],rate:[3,7,8],rather:2,ratio:[2,7,9],ratio_subscrib:9,ratiosubscrib:[2,9],read:2,realli:2,reason:2,receiv:[1,2,9],receive_messag:2,recept:2,recip:3,recomput:5,record:2,rectangular:5,recurs:[1,4,9],redraw:5,reduc:[3,9],refactor:5,refer:9,regardless:2,regist:[1,2,3,9],register_data_top:1,register_forward_hook:9,register_meta_top:1,register_subscrib:[1,9],rel:1,relai:[1,2,9],relat:9,releas:[1,9],relev:2,reli:[1,9],reliabl:5,relu:[1,3,9],remov:9,replac:[4,5,8,9],report:6,repres:9,represent:[4,8],reqorgan:5,requir:9,requires_grad:4,reset:[1,2],reshap:2,resnet:3,respect:1,restart:8,retriev:4,reus:2,ridden:9,root:4,round:2,round_idx:2,routin:2,run:[2,3,7,8,9],runtimeerror:[1,2],same:[1,4,9],sampl:2,scalar:[2,5,9],scale1:9,scale2:9,scale:5,schedul:8,search:6,second:[2,9],see:[1,2,9],seed:7,seen:2,select:2,self:[4,8,9],sender:9,sens:2,sequenc:[1,2,4,8],sequenti:[1,4,9],set:[1,2,3,4,5,7,8,9],set_loss:[1,9],set_model:[1,8,9],set_schedul:8,setup:9,setuptool:9,sever:[2,9],shape:[5,8],share:4,ship:9,shorthand:1,should:[1,2,3,4,5,9],signal:1,signific:9,silent:3,simpl:9,simplifi:7,sinc:[2,3,4,5],singl:[1,2,9],singular:2,size:[2,3,5,7,8,9],skip:4,slower:3,small:4,small_input:3,softwar:4,some:[1,8,9],somewhat:[4,9],sourc:[1,2,3,4,5,7,8,9],space:[4,5],span:2,special:2,specialis:9,specif:[1,2,9],specifi:[1,4,8],spectral:[2,7],spectralnormsubscrib:2,sphinx:7,spotti:4,ssh:5,stabil:7,start:2,statist:2,step:[1,2,5,9],still:1,storag:4,str:[1,2,4,5,7,8],straightforward:9,stream:2,stride:[3,9],string:[1,2,4,8],stuff:2,stupidli:9,sub:1,subclass:[1,2,3,9],sublot:5,submodul:[0,6],subpackag:6,subplot:5,subsampl:[2,7,9],subscrib:[0,1,5,6,7,8],subscript:[1,2,9],subset:2,substanti:7,success:4,sum:2,summari:2,summarywrit:5,sumsubscrib:2,superclass:9,superflu:4,supervis:[1,9],support:[1,4],sure:[2,9],surviv:[3,9],svd:2,synchronis:9,synchronizedsubscript:[1,2,9],system:9,tag:[1,2,9],take:[3,9],taken:2,target:8,tbbackend:5,tensor:[1,2,4,5,9],tensor_to_numba:4,tensorboard:[5,7,9],tensorboardx:5,tenth:2,test:[1,2,4,7],test_accuraci:2,test_transform:4,testaccuracysubscrib:2,thei:[2,3,9],them:[1,2,3,5,9],therefor:[2,9],thesi:9,thi:[1,2,3,4,5,7,8,9],think:8,those:[1,9],three:[1,9],through:[2,8,9],throughout:9,tied:[1,9],time:[1,2,7,9],titl:[2,5,9],togeth:2,top:2,topic:[1,2,7,9],torch:[1,2,3,4,5,8,9],torchvis:[4,9],total:2,totensor:4,track:[1,4,9],tracker:[7,9],train:[1,2,4,6,7,9],train_accuraci:2,train_batch:8,train_step:[1,9],train_transform:4,trainaccuracysubscrib:2,trainer:[2,8],transform:4,transpar:9,travers:[1,4,8],treatment:2,tree:[1,4,8],trigger:1,tupl:[1,2,3,4,5,8],turn:[1,9],twice:[2,5],two:[2,9],type:[1,2,3,4,5,7,8,9],typestr:4,under:9,underli:4,underscor:5,unfortun:9,uninitialis:8,unless:1,unmodifi:1,unqualifi:7,unregist:1,unstabl:9,unus:[7,9],updat:[1,7],updatablehistogram:5,upon:2,usag:7,use:[1,2,3,5,7,8,9],useabl:5,used:[1,2,4,8,9],useful:[2,4,9],uses:[2,9],using:2,usr:9,util:[0,1,2,6,8],valid:2,valu:[2,4,8],valueerror:[1,2,4,9],varianc:2,variancesubscrib:2,verbos:7,veri:9,version:[2,9],vertic:5,vgg:3,via:[5,8],view:9,virtu:9,virtualenv:9,visual:[0,2,6],visuali:5,visualis:[5,7,9],vital:[2,9],w_out:[3,9],wai:2,want:[2,9],wasn:1,weight:[1,2,4,9],weights_:9,well:9,were:2,what:[1,2,3,9],whatev:[2,9],when:[1,2],whenev:9,where:2,whether:[1,2,9],which:[1,2,3,4,5,7,8,9],whitespac:5,wide:[5,9],width:[3,9],wire:1,wish:2,within:3,without:9,work:[2,4,6,9],world:9,worri:9,would:[2,9],wrap:8,wrapper:9,write:9,wrt:1,xlabel:[2,5,9],yet:[1,2],yield:4,ylabel:[2,5,9],ylim:[2,5,9],you:[2,3,9],your:9,yourself:9,yoursubscrib:9},titles:["ikkuna","ikkuna.export","ikkuna.export.subscriber","ikkuna.models","ikkuna.utils","ikkuna.visualization","Ikkuna","main program","train","User Guide"],titleterms:{"export":[1,2],"new":9,api:6,argument:7,content:[0,1,2,3,4,5,8,9],creat:9,detail:9,guid:[6,9],ikkuna:[0,1,2,3,4,5,6],indic:6,instal:9,introduct:9,issu:9,librari:9,main:7,messag:1,model:3,modul:[1,2,3,4,5,8],name:7,numba:4,prerequisit:9,program:7,quickstart:9,refer:6,report:9,submodul:[1,4],subpackag:[0,1],subscrib:[2,9],tabl:6,train:8,user:[6,9],util:4,visual:5}})