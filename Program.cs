using System;
using System.Collections.Generic;
using System.Linq;
using System.Data;
using System.Configuration;
using System.Collections;
using System.Web;
using System.Net;
using System.Text;
using System.IO;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace SPMR_TMM  //只考虑两个因素 POI情感相似性  user和POI之间的距离
{
    class Program
    {
        //public static string yelp_test_path = @"E:\data\Yelp\data1\test_restaurants-TMC\";
        //public static string yelp_test_all_cate_path = @"E:\data\Yelp\data1\test_restaurants-TMC\";
        //public static string yelp_user_path = @"E:\data\Yelp\users_all\";


        public static int K = 10;//Q,P潜在特征向量维数(即参数个数)
        public static int iteration_count = 50;//Q,P潜在特征向量维数(即参数个数)
        public const double P = 0.0005;//步长  0.001 乌鲁木齐  其他0.0005
        public static double rm = 0;//预测补偿值(训练数据中用户评论均值)
        public static double beta = 0.1;//目标函数系数    0.1
        public static double garma = 0;//POI    10 
        public static double suv = 0;//friends      0.1
        public static double lui = 1;//目标函数系数Lui      1
        public static double lii = 0;//目标函数系数Lii
        public static double yita = 0;//目标函数系数user-item similarity10
        public static double lanwta_P = 0.1;//目标函数系数3
        public static double lanwta_Q = 0.1;//目标函数系数19
        public const double EARTH_RADIUS = 6378.137;//地球半径

        public static double t = 0.1;
        public static int consider_garma = 0;
        public static int consider_suv = 0;
        public static int consider_lui = 1;

        public static int test_data_index = 1;
        public static string category = "长文\\北京";
        //public static string CF_model = "Gauss2";
        //public static string CF_Lui = category + CF_model;
        //public static string CF_Luu = category + CF_model;
        public static string yelp_test_result_path = @"SPM_stepdown\\" + category + "\\testdata" + test_data_index + "\\";
        public static string yelp_test_path = @"D:\zgs\POI推荐\数据集\" + category + "\\";
        public static string yelp_test_all_cate_path = @"D:\zgs\POI推荐\数据集\" + category + "\\";


        static void Main(string[] args)
        {

            double ss = Program.P;
            //test_data_index = index;
            yelp_test_result_path = @"SPM_stepdown\\" + category + "\\testdata" + test_data_index + "\\";
            if (!Directory.Exists(yelp_test_result_path + "FriendsCircle"))
            {
                Directory.CreateDirectory(yelp_test_result_path + "FriendsCircle");
            }
            if (!Directory.Exists(yelp_test_result_path + "P"))
            {
                Directory.CreateDirectory(yelp_test_result_path + "P");
            }
            if (!Directory.Exists(yelp_test_result_path + "Q"))
            {
                Directory.CreateDirectory(yelp_test_result_path + "Q");
            }

            DateTime dt1 = DateTime.Now;
            ThreadPool.SetMaxThreads(12, 12);
            double error = 0;//误差值,小于一定值则结束训练
            int iter_num = 0;//迭代次数计数器
            List<string> u_i_ratings = new List<string>();
            List<string> u_i_rating_test = new List<string>();
            List<string> users_id = new List<string>();//resturant类下的所有用户
            List<string> items_id = new List<string>();//resturant类下的所有item
            #region//读入所有用户及item
            FileStream fs1 = new FileStream(yelp_test_path + "users.txt", FileMode.Open);
            StreamReader sr = new StreamReader(fs1);
            string[] users = Regex.Split(sr.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
            sr.Close();
            fs1.Close();
            for (int i = 0; i < users.Length - 1; i++)
            {
                users_id.Add(users[i].Split(';')[0]);
            }
            fs1 = new FileStream(yelp_test_path + "pois.txt", FileMode.Open);
            sr = new StreamReader(fs1);
            string[] items = Regex.Split(sr.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
            sr.Close();
            fs1.Close();
            for (int i = 0; i < items.Length - 1; i++)
            {
                items_id.Add(items[i].Split(';')[0]);
            }
            #endregion

            double[][] U_I_R = new double[users_id.Count][];
            for (int i = 0; i < users_id.Count; i++)
                U_I_R[i] = new double[items_id.Count];

            #region//读入训练数据
            string u_i_rating_training_file = yelp_test_path + "training.txt";
            fs1 = new FileStream(u_i_rating_training_file, FileMode.Open);
            sr = new StreamReader(fs1);
            string[] str_line = Regex.Split(sr.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
            sr.Close();
            fs1.Close();
            for (int i = 0; i < str_line.Length - 1; i++)
                u_i_ratings.Add(str_line[i]);
            #endregion

            #region//读入测试数据
            string u_i_rating_test_file = yelp_test_path + "test.txt";
            FileStream fs2 = new FileStream(u_i_rating_test_file, FileMode.Open);
            StreamReader sr2 = new StreamReader(fs2);
            string[] str_line2 = Regex.Split(sr2.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
            sr2.Close();
            fs2.Close();
            for (int i = 0; i < str_line2.Length - 1; i++)
                u_i_rating_test.Add(str_line2[i]);
            #endregion

            #region//建立用户-商品评论矩阵,并计算rm
            Parallel.For(0, u_i_ratings.Count, i =>
            //for (int i = 0; i < u_i_ratings.Count; i++)
            {
                lock (syn)
                {
                    string[] u_i_ids = u_i_ratings[i].Split(';');
                    string user_id = u_i_ids[0];
                    string item_id = u_i_ids[1];
                    //U_I_R[users_id.IndexOf(user_id)][items_id.IndexOf(item_id)] = Convert.ToDouble(u_i_ids[2]);
                    U_I_R[users_id.IndexOf(user_id)][items_id.IndexOf(item_id)] = 1 / (1 + 1 / (Convert.ToDouble(u_i_ids[2])));////////
                    //rm += Convert.ToDouble(u_i_ids[2]);
                    rm += 1 / (1 + 1 / (Convert.ToDouble(u_i_ids[2])));////////
                }
            });
            rm /= u_i_ratings.Count;
            #endregion

            double[][] Lui = new double[users_id.Count][];
            for (int i = 0; i < users_id.Count; i++)
                Lui[i] = new double[items_id.Count];

            #region//计算并保存Lui//user与item之间的距离因素  距离近 则评分高  距离远 则评分低
            //Lui_Statistic(u_i_ratings, users_id, items_id, Lui);
            //FileStream fsww = new FileStream("jaodjfo.txt", FileMode.Create);
            //StreamWriter sww = new StreamWriter(fsww);
            //sww.Close();
            //fsww.Close();
            //if (!Directory.Exists(yelp_test_all_cate_path + "testdata" + test_data_index + "\\Lui"))
            //{
            //    Directory.CreateDirectory(yelp_test_all_cate_path + "testdata" + test_data_index + "\\Lui");
            //}
            //for (int i = 0; i < users_id.Count; i++)
            //{
            //    fsww = new FileStream(yelp_test_all_cate_path + "testdata" + test_data_index + "\\Lui\\" + users_id[i] + ".txt", FileMode.Create);
            //    sww = new StreamWriter(fsww);
            //    for (int j = 0; j < Lui[i].Length; j++)
            //    {
            //        sww.WriteLine(Lui[i][j]);
            //    }
            //    sww.Close();
            //    fsww.Close();
            //}
            //////read Lui
            for (int i = 0; i < users_id.Count; i++)
            {
                fs1 = new FileStream(yelp_test_all_cate_path + "testdata" + test_data_index + "\\Lui\\" + users_id[i] + ".txt", FileMode.Open);
                sr = new StreamReader(fs1);
                string[] content = Regex.Split(sr.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
                sr.Close();
                fs1.Close();
                Lui[i] = new double[content.Length - 1];
                Parallel.For(0, content.Length - 1, j =>
                //for (int j = 0; j < content.Length-1; j++)
                {
                    Lui[i][j] = Convert.ToDouble(content[j]);
                });
            }
            #endregion

            double[][] Eij = new double[items_id.Count][];//POI趣相似度矩阵
            #region//计算并保存// POI之间的情感相似性！

            //SimiUI_new(u_i_ratings, items_id, Eij);
            //if (!Directory.Exists(yelp_test_all_cate_path + "根据distance抽取训练测试集\\testdata" + test_data_index + "\\Eij"))
            //{
            //    Directory.CreateDirectory(yelp_test_all_cate_path + "根据distance抽取训练测试集\\testdata" + test_data_index + "\\Eij");
            //}
            //for (int i = 0; i < items_id.Count; i++)
            //{
            //    fsww = new FileStream(yelp_test_all_cate_path + "根据distance抽取训练测试集\\testdata" + test_data_index + "\\Eij\\" + items_id[i] + ".txt", FileMode.Create);
            //    sww = new StreamWriter(fsww);
            //    for (int j = 0; j < Eij[i].Length; j++)
            //    {
            //        sww.WriteLine(Eij[i][j]);
            //    }
            //    sww.Close();
            //    fsww.Close();
            //}
            ////////////read
            for (int i = 0; i < items_id.Count; i++)
            {
                fs1 = new FileStream(yelp_test_all_cate_path + "根据distance抽取训练测试集\\testdata" + test_data_index + "\\Eij\\" + items_id[i] + ".txt", FileMode.Open);
                sr = new StreamReader(fs1);
                string[] content = Regex.Split(sr.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
                sr.Close();
                fs1.Close();
                Eij[i] = new double[content.Length - 1];
                Parallel.For(0, content.Length - 1, j =>
                //for (int j = 0; j < content.Length-1; j++)
                {
                    Eij[i][j] = Convert.ToDouble(content[j]);
                });
            }
            #endregion

            double[][] Suv = new double[users_id.Count][];//用户趣相似度矩阵
            
            List<List<int>> Friends_id = new List<List<int>>();
            

            //训练目标即得到Q,P
            double[][] Q = new double[users_id.Count][];//resturant类的用户潜在特征向量
            double[][] P = new double[items_id.Count][];//resturant类的item潜在特征向量
            #region//对Q,P赋正态分布初值
            Random rand = new Random();

            double p = 0;//==============================================================================================================
            double q = 0;//==============================================================================================================
            FileStream read_q = new FileStream(@"D:\zgs\POI推荐\数据集\Q.txt", FileMode.Open);
            StreamReader readq = new StreamReader(read_q);
            for (int i = 0; i < users_id.Count; i++)
            {
                Q[i] = new double[K];
                for (int j = 0; j < K; j++)
                {
                    // Q[i][j] = 0.0001;// NormalDistribution(rand);
                    Q[i][j] = Convert.ToDouble(readq.ReadLine()) * t;
                    q = q + Q[i][j];//==============================================================================================================

                }
            }
            readq.Close();
            read_q.Close();
            FileStream read_p = new FileStream(@"D:\zgs\POI推荐\数据集\P.txt", FileMode.Open);
            StreamReader readp = new StreamReader(read_p);
            for (int i = 0; i < items_id.Count; i++)
            {
                P[i] = new double[K];
                for (int j = 0; j < K; j++)
                {
                    //P[i][j] = 0.0001;// NormalDistribution(rand);
                    P[i][j] = Convert.ToDouble(readp.ReadLine()) * t;
                    p = p + P[i][j];//==============================================================================================================

                }
            }
            readp.Close();
            read_p.Close();
            #endregion

            p = p / (double)items_id.Count;//==============================================================================================================
            q = q / (double)users_id.Count;//==============================================================================================================

            double[] BU = new double[users_id.Count];
            double[] BI = new double[items_id.Count];
            string save_error_files = yelp_test_result_path + "error_0.01lanwta_" + garma + "garma_" + beta + "bate_200iterative.txt";
            FileStream fsw3 = new FileStream(save_error_files, FileMode.Create);
            StreamWriter sw3 = new StreamWriter(fsw3);

            error = ObjectiveFuncation(u_i_ratings, Q, P, BU, BI, Eij, Lui,Suv, users_id, items_id, Friends_id, U_I_R);
            error = 50;
            sw3.WriteLine(iter_num + " " + error);
            while (error > 0.005 && iter_num < iteration_count)
            {
                double[][] gu = new double[users_id.Count][];
                double[][] gi = new double[items_id.Count][];
                Parallel.For(0, users_id.Count, i =>////////////////////////////////////////////////////////////////////////
                {
                    gu[i] = new double[K];
                    Gradient_u(i, U_I_R, Q, P, BU, BI, Eij, Lui, Suv, Friends_id, users_id, items_id, gu[i]);

                });

                Parallel.For(0, items_id.Count, i =>////////////////////////////////////////////////////////////////////////
                {
                    gi[i] = new double[K];
                    Gradient_i(i, U_I_R, Q, P, BU, BI, Eij, Lui, users_id, items_id, gi[i]);
                });
                if (yelp_test_result_path.Split('\\')[0].Contains("SPM_stepdown"))
                {
                    ss = ss * 0.9;
                }

                Parallel.For(0, users_id.Count, i =>////////////////////////////////////////////////////////////////////////
                //Parallel.For(0, 200, (j) =>
                //for (int i = 0; i < users_id.Count; i++)
                {
                    for (int l = 0; l < K; l++)
                        Q[i][l] = Q[i][l] - ss * gu[i][l];

                });
                Parallel.For(0, items_id.Count, i =>////////////////////////////////////////////////////////////////////////
                {
                    for (int l = 0; l < K; l++)
                        P[i][l] = P[i][l] - ss * gi[i][l];

                });
                // ss = ss * 0.9;
                for (int i = 0; i < users_id.Count; i++)
                {
                    Parallel.For(0, items_id.Count, j =>
                    //for (int j = 0; j < items_id.Count; j++)
                    {
                        lock (syn)
                        {
                            if (U_I_R[i][j] != 0)
                            {
                                double predicted = r_ui(rm, BU[i], BI[j], Q[i], P[j]);
                                BU[i] = BU[i] + ss * (U_I_R[i][j] - predicted - beta * BU[i]);
                                BI[j] = BI[j] + ss * (U_I_R[i][j] - predicted - beta * BI[j]);
                            }
                        }
                    });
                }

                Console.WriteLine("iter_num:" + iter_num);
                Console.WriteLine("error:" + error);
                iter_num++;
                sw3.WriteLine("step: " + ss);
                sw3.WriteLine(iter_num + " " + error);
                if (iter_num % 1 == 0)
                {
                    #region//保存结果
                    /*
                    double p1 = 0;//==============================================================================================================
                    double q1 = 0;//==============================================================================================================
                    for (int i = 0; i < users_id.Count; i++)
                    {
                        string save_Q_files = yelp_test_result_path + "Q\\" + users_id[i] + ".txt";
                        FileStream fsw5 = new FileStream(save_Q_files, FileMode.Create);
                        StreamWriter sw5 = new StreamWriter(fsw5);
                        for (int j = 0; j < K; j++)
                        {
                            sw5.WriteLine(Q[i][j]);
                            q1 = q1 + Q[i][j];//==============================================================================================================
                        }
                        sw5.Close();
                        fsw5.Close();
                    }


                    for (int i = 0; i < items_id.Count; i++)
                    {
                        string save_P_files = yelp_test_result_path + "P\\" + items_id[i] + ".txt";
                        FileStream fsw4 = new FileStream(save_P_files, FileMode.Create);
                        StreamWriter sw4 = new StreamWriter(fsw4);
                        for (int j = 0; j < K; j++)
                        {
                            sw4.WriteLine(P[i][j]);
                            p1 = p1 + P[i][j];//==============================================================================================================
                        }
                        sw4.Close();
                        fsw4.Close();
                    }

                    p1 = p1 / (double)items_id.Count;//==============================================================================================================
                    q1 = q1 / (double)users_id.Count;//==============================================================================================================

                    Console.WriteLine("p:" + p);
                    Console.WriteLine("p1:" + p1);
                    Console.WriteLine("q:" + q);
                    Console.WriteLine("q1:" + q1);
                    */
                    #endregion

                    #region//rmse
                    double rmse = 0;

                    #region//rating predict
                    double[] u_i_r_t = new double[u_i_rating_test.Count];//用户实际评分
                    double[] u_i_rp_t = new double[u_i_rating_test.Count];//预测评分值
                    //FileStream fsw = new FileStream(yelp_test_result_path + "test_ratings" + lanwta_P + "P_" + beta + "beta_" + garma + "garma_" + lui + "lui_" + iter_num + "iteration_" + Program.P + "step.txt", FileMode.Create);
                    //StreamWriter sw = new StreamWriter(fsw);
                    for (int i = 0; i < u_i_rating_test.Count; i++)
                    {
                        string[] u_i_ids = u_i_rating_test[i].Split(';');
                        //u_i_r_t[i] = Convert.ToDouble(u_i_ids[2]);
                        u_i_r_t[i] = 1 / (1 + 1 / (Convert.ToDouble(u_i_ids[2])));/////////////////////////////
                        u_i_rp_t[i] = rm + MatrixMulti(Q[users_id.IndexOf(u_i_ids[0])], P[items_id.IndexOf(u_i_ids[1])]) + BU[users_id.IndexOf(u_i_ids[0])] + BI[items_id.IndexOf(u_i_ids[1])];
                        //u_i_rp_t[i] =Math.Round( rm + MatrixMulti(Q[users_id.IndexOf(u_i_ids[0])], P[items_id.IndexOf(u_i_ids[1])]));//==================================================================================================================================
                        //sw.WriteLine(u_i_ids[0] + "\t" + u_i_ids[1] + "\t" + u_i_ids[2] + "\t" + u_i_rp_t[i]);
                        //sw.WriteLine(u_i_ids[0] + "\t" + u_i_ids[1] + "\t" + 1 / (1 + 1 / (Convert.ToDouble(u_i_ids[2]))) + "\t" + u_i_rp_t[i]);
                    }
                    //sw.Close();
                    //fsw.Close();
                    #endregion
                    double sum1 = 0;//预测误差
                    #region//sum1
                    for (int i = 0; i < u_i_r_t.Length; i++)
                    {
                        sum1 = sum1 + (u_i_r_t[i] - u_i_rp_t[i]) * (u_i_r_t[i] - u_i_rp_t[i]);
                    }
                    #endregion
                    rmse = Math.Sqrt(sum1 / u_i_rating_test.Count);
                    #endregion


                    #region//mae
                    double mae = 0;
                    for (int i = 0; i < u_i_r_t.Length; i++)
                        mae += Math.Abs(u_i_r_t[i] - u_i_rp_t[i]);
                    mae /= u_i_rating_test.Count;
                    #endregion


                    #region///计算Precision and Recall
                    double[] Precision = new double[items_id.Count];
                    double[] Recall = new double[items_id.Count];
                    double[] MAP = new double[2];
                    Precision_Recall(u_i_rating_test, users_id, items_id, U_I_R, Q, P, BU, BI, u_i_rp_t, Precision, Recall, MAP);


                    #endregion

                    string save_result_files = yelp_test_result_path + "result_" + lanwta_P + "P_" + lanwta_Q + "Q_" + beta + "beta_" + garma + "garma_" + suv + "suv_" + lui + "lui_" + iter_num + "iteration_" + Program.P + "step_" + t + "t" + ".txt";

                    //string save_result_files = yelp_test_result_path + "result_" + CF_Lui + "_" + CF_Luu + "_" + garma + "garma_" + beta + "beta_" + yita + "yita_" + lui + "lui_" + luu + "luu_" + lanwta_P + "P_" + lanwta_Q + "Q_" +"itera"+ ".txt";
                    FileStream fsw6 = new FileStream(save_result_files, FileMode.Create);
                    StreamWriter sw6 = new StreamWriter(fsw6);
                    sw6.WriteLine("users_num" + users_id.Count);
                    sw6.WriteLine("iterms_num" + items_id.Count);
                    sw6.WriteLine("rating_num" + u_i_ratings.Count);
                    sw6.WriteLine("rm:" + rm);
                    sw6.WriteLine("rmse:" + rmse);
                    sw6.WriteLine("mae:" + mae);
                    sw6.WriteLine("p:" + p);
                    sw6.WriteLine("q:" + q);
                    //sw6.WriteLine("p1:" + p1);
                    //sw6.WriteLine("q1:" + q1);
                    DateTime dt2 = DateTime.Now;
                    sw6.WriteLine("平行运算运行时长：{0}小时。", (dt2 - dt1).TotalHours);
                    Console.WriteLine("平行运算运行时长：{0}小时。", (dt2 - dt1).TotalHours);
                    sw6.Close();
                    fsw6.Close();
                    if (!Directory.Exists(yelp_test_result_path + "PR"))
                    {
                        Directory.CreateDirectory(yelp_test_result_path + "PR");
                    }
                    fsw6 = new FileStream(yelp_test_result_path + "PR\\" + "result_" + lanwta_P + "P_" + beta + "beta_" + garma + "garma_" + lui + "lui_" + iter_num + "iteration_" + Program.P + "step" + ".txt", FileMode.Create);
                    sw6 = new StreamWriter(fsw6);
                    for (int i = 0; i < Precision.Length; i++)
                    {
                        sw6.WriteLine(Precision[i] + "\t" + Recall[i]);
                    }

                    sw6.WriteLine();
                    sw6.WriteLine("MAP:" + MAP[0]);
                    sw6.Close();
                    fsw6.Close();

                }


            }
            sw3.Close();
            fsw3.Close();

        }
        public static double r_ui(double miu, double B_u, double B_i, double[] P_u, double[] Q_i)
        {
            double res = 0;
            res = miu + B_u + B_i + MatrixMulti(P_u, Q_i);
            return res;
        }

        public static void Gradient_i(int item_id, double[][] U_I_R, double[][] Q, double[][] P, double[] Bu, double[] Bi, double[][] Eij, double[][] Lui, List<string> users_id, List<string> items_id, double[] gi)
        {

            double[] gi1 = new double[K];
            double[] gi2 = new double[K];
            double[] gi3 = new double[K];

            double[] gi5 = new double[K];

            #region//gi1
            List<double> a1 = new List<double>();
            List<int> user_index = new List<int>();
            for (int i = 0; i < U_I_R.Length; i++)
            {
                if (U_I_R[i][item_id] != 0)
                {
                    double a = 0;
                    a = rm + MatrixMulti(Q[i], P[item_id]) + Bu[i] + Bi[item_id] - U_I_R[i][item_id];
                    a1.Add(a);
                    user_index.Add(i);
                }
            }
            //Parallel.For(0, K, (l) =>////////////////////////////////////////////////////////////////////////////////////////////////////
            for (int l = 0; l < K; l++)
            {
                for (int i = 0; i < a1.Count; i++)
                {
                    gi1[l] = gi1[l] + a1[i] * Q[user_index[i]][l];
                }
                gi1[l] = gi1[l] + lanwta_P * P[item_id][l];
            }//);
            #endregion

            #region//gi2
            /*
            List<double> a2 = new List<double>();
            for (int i = 0; i < U_I_R.Length; i++)
            {
                if (U_I_R[i][item_id] != 0)
                {
                    double a = 0;
                    a = (MatrixMulti(Q[i], P[item_id]) - QPS[i][item_id]) * Nuc[i] * yita;
                    a2.Add(a);
                }
            }
            for (int l = 0; l < K; l++)
            {
                for (int i = 0; i < a2.Count; i++)
                {
                    gi2[l] = gi2[l] + a2[i] * Q[user_index[i]][l];
                }
            }
            */
            #endregion


            #region//gi3   /////////////////////////////////////////////////////Lui
            if (consider_lui == 1)
            {
                List<double> a3 = new List<double>();
                double sum = 0;
                for (int i = 0; i < U_I_R.Length; i++)
                {
                    if (U_I_R[i][item_id] != 0)
                    {
                        sum = sum + Lui[i][item_id];
                    }
                }
                for (int i = 0; i < U_I_R.Length; i++)
                {
                    if (U_I_R[i][item_id] != 0)
                    {
                        double a = 0;
                        a = (MatrixMulti(Q[i], P[item_id]) - Lui[i][item_id] / sum) * lui;
                        a3.Add(a);
                    }
                }
                //Parallel.For(0, K, (l) =>/////////////////////
                for (int l = 0; l < K; l++)
                {
                    for (int i = 0; i < a3.Count; i++)
                    {
                        gi3[l] = gi3[l] + a3[i] * Q[user_index[i]][l];
                    }
                }//);
            }
            #endregion

            #region//gi5 加入POI相似度的影响力因素
            if (consider_garma == 1)
            {
                //加入用户兴趣相似度的影响力因素
                double[] Qiv = new double[K];
                //Parallel.For(0, K, (l) =>///////////////////////////////////////////////////////////////////////////////////
                for (int l = 0; l < K; l++)
                {
                    for (int j = 0; j < Eij[item_id].Length; j++)
                    {
                        Qiv[l] = Qiv[l] + Eij[item_id][j] * P[j][l];
                    }
                }//);
                //Parallel.For(0, K, (l) =>///////////////////////////////////////////////////////////////////////////////////
                for (int l = 0; l < K; l++)
                {
                    gi5[l] = garma * (P[item_id][l] - Qiv[l]);
                }//);
            #endregion

                #region//gu5
                //Parallel.For(0, K, (m) =>///////////////////////////////////////////////////////////////////////////////////

                for (int i = 0; i < items_id.Count; i++)
                {
                    int v_id = i;
                    //List<int> w_in_c = new List<int>();//该类别里v的好友w
                    //FriendCirle(v_in_c[i], users_id, w_in_c);

                    double Ivu = Eij[v_id][item_id];
                    double[] Qw = new double[K];
                    //Parallel.For(0, K, (l) =>///////////////////////////////////////////////////////////////////////////////////
                    for (int l = 0; l < K; l++)
                    {
                        for (int j = 0; j < items_id.Count; j++)
                        {
                            Qw[l] = Qw[l] + Eij[v_id][j] * P[j][l];
                        }
                        gi5[l] = gi5[l] + garma * Ivu * (P[v_id][l] - Qw[l]);
                    }//);

                }
            }
                #endregion
            #region//gi5  /////////////////////////////////////////Lii
            /*
            ////加入用户Lii的影响力因素
            double sum1 = 0;
            double somevalue = 0;

            List<int> J = new List<int>();
            for (int i = 0; i < U_I_R.Length; i++)
            {
                if (U_I_R[i][item_id] != 0)
                {
                    J = new List<int>();
                    sum1 = 0;
                    somevalue = 0;
                    //max = 1;
                    for (int j = 0; j < U_I_R[i].Length; j++)
                    {
                        if (U_I_R[i][j] != 0)
                        {
                            sum1 = sum1 + Lui[i][j];
                            J.Add(j);
                        }
                    }

                    for (int a = 0; a < J.Count; a++)
                    {
                        somevalue = somevalue + (Lui[i][J[a]] / sum1) * MatrixMulti(Q[i], P[J[a]]);// *Lii[item_id][J[a]] / max;
                    }
                    for (int l = 0; l < K; l++)
                    {
                        gi5[l] = gi5[l] + (MatrixMulti(Q[i], P[item_id]) - somevalue) * Q[i][l] * lii;
                    }//);
                }
            }
            */
            #endregion


            for (int i = 0; i < K; i++)
            {
                gi[i] = gi1[i] + gi2[i] + gi3[i] + gi5[i];
            }//);
        }

        public static void Gradient_u(int user_id, double[][] U_I_R, double[][] Q, double[][] P, double[] Bu, double[] Bi, double[][] Eij, double[][] Lui, double[][] Suv, List<List<int>> Friends_id, List<string> users_id, List<string> items_id, double[] gu)
        {
            double[] gu1 = new double[K];
            double[] gu2 = new double[K];
            double[] gu3 = new double[K];
            double[] gu4 = new double[K];
            double[] gu5 = new double[K];
            double[] gu6 = new double[K];
            double[] gu7 = new double[K];
            double[] gu8 = new double[K];
            double[] gu9 = new double[K];
            double[] gu10 = new double[K];

            #region//gu1

            List<double> a1 = new List<double>();
            List<int> item_index = new List<int>();
            for (int i = 0; i < U_I_R[user_id].Length; i++)
            {
                if (U_I_R[user_id][i] != 0)
                {
                    double a = 0;
                    a = rm + MatrixMulti(Q[user_id], P[i]) - U_I_R[user_id][i] + Bu[user_id] + Bi[i];
                    a1.Add(a);
                    item_index.Add(i);
                }
            }
            //Parallel.For(0, K, (l) =>///////////////////////////////////////////////////////////////////////////////////
            for (int l = 0; l < K; l++)
            {
                for (int i = 0; i < a1.Count; i++)
                {
                    gu1[l] = gu1[l] + a1[i] * P[item_index[i]][l];
                }
                gu2[l] = lanwta_Q * Q[user_id][l];
            }//);
            #endregion

            #region//gu2
            if (consider_suv == 1)
            {
                //加入用户trusters的影响力因素
                double[] Qv = new double[K];
                //Parallel.For(0, K, (l) =>///////////////////////////////////////////////////////////////////////////////////
                for (int l = 0; l < K; l++)
                {
                    for (int j = 0; j < Suv[user_id].Length; j++)
                    {
                        Qv[l] = Qv[l] + Suv[user_id][j] * Q[Friends_id[user_id][j]][l];
                    }
                }//);
                //Parallel.For(0, K, (l) =>///////////////////////////////////////////////////////////////////////////////////
                for (int l = 0; l < K; l++)
                {
                    gu2[l] = gu2[l] + suv * (Q[user_id][l] - Qv[l]);
                }//);
            #endregion

                #region//gu3

                //Parallel.For(0, K, (m) =>///////////////////////////////////////////////////////////////////////////////////

                for (int i = 0; i < Friends_id[user_id].Count; i++)
                {
                    int v_id = Friends_id[user_id][i];
                    double Svu = Suv[v_id][Friends_id[v_id].IndexOf(user_id)];
                    double[] Qw = new double[K];
                    //Parallel.For(0, K, (l) =>///////////////////////////////////////////////////////////////////////////////////
                    for (int l = 0; l < K; l++)
                    {
                        for (int j = 0; j < Friends_id[v_id].Count; j++)
                        {
                            Qw[l] = Qw[l] + Suv[v_id][j] * Q[Friends_id[v_id][j]][l];
                        }
                        gu3[l] = gu3[l] + suv * Svu * (Q[v_id][l] - Qw[l]);
                    }

                }
            }
                #endregion


            #region//gu4 gu5 加入用户兴趣相似度的影响力因素
            /*
            //加入用户兴趣相似度的影响力因素
            double[] Qiv = new double[K];
            //Parallel.For(0, K, (l) =>///////////////////////////////////////////////////////////////////////////////////
            for (int l = 0; l < K; l++)
            {
                for (int j = 0; j < Iuv[user_id].Length; j++)
                {
                    Qiv[l] = Qiv[l] + Iuv[user_id][j] * Q[FriendsCircle[user_id][j]][l];
                }
            }//);
            //Parallel.For(0, K, (l) =>///////////////////////////////////////////////////////////////////////////////////
            for (int l = 0; l < K; l++)
            {
                gu4[l] = garma * (Q[user_id][l] - Qiv[l]);
            }//);
            #endregion

            #region//gu5
            //Parallel.For(0, K, (m) =>///////////////////////////////////////////////////////////////////////////////////

            for (int i = 0; i < FriendsCircle[user_id].Length; i++)
            {
                int v_id = FriendsCircle[user_id][i];
                List<int> w_in_c = new List<int>();//该类别里v的好友w
                //FriendCirle(v_in_c[i], users_id, w_in_c);
                for (int j = 0; j < FriendsCircle[v_id].Length; j++)
                {
                    w_in_c.Add(FriendsCircle[v_id][j]);
                }
                if (w_in_c.Contains(user_id))//u也是v的truster
                {
                    double Ivu = Iuv[v_id][w_in_c.IndexOf(user_id)];
                    double[] Qw = new double[K];
                    //Parallel.For(0, K, (l) =>///////////////////////////////////////////////////////////////////////////////////
                    for (int l = 0; l < K; l++)
                    {
                        for (int j = 0; j < w_in_c.Count; j++)
                        {
                            Qw[l] = Qw[l] + Iuv[v_id][j] * Q[w_in_c[j]][l];
                        }
                        gu5[l] = gu5[l] + garma * Ivu * (Q[v_id][l] - Qw[l]);
                    }//);
                    //for (int m = 0; m < K; m++)
                    //{

                    //}
                }
            }
            */

            #endregion

            #region//gu7/////////////////////////////////////////////////////Lui
            if (consider_lui == 1)
            {
                List<double> a3 = new List<double>();
                double sum = 0;
                for (int i = 0; i < U_I_R[user_id].Length; i++)
                {
                    if (U_I_R[user_id][i] != 0)
                    {
                        sum = sum + Lui[user_id][i];
                    }
                }
                for (int i = 0; i < U_I_R[user_id].Length; i++)
                {
                    if (U_I_R[user_id][i] != 0)
                    {
                        double a = 0;
                        a = (MatrixMulti(Q[user_id], P[i]) - Lui[user_id][i] / sum) * lui;
                        a3.Add(a);
                    }
                }
                //Parallel.For(0, K, (l) =>////////////////
                for (int l = 0; l < K; l++)
                {
                    for (int i = 0; i < a3.Count; i++)
                    {
                        gu7[l] = gu7[l] + a3[i] * P[item_index[i]][l];
                    }
                }//);
            }
            #endregion



            for (int l = 0; l < K; l++)
            {
                gu[l] = gu1[l] + gu2[l] - gu3[l] + gu4[l] - gu5[l] + gu6[l] + gu7[l] + gu8[l] - gu9[l] + gu10[l];
            }//);
        }
        private static object syn = new object();
        public static double ObjectiveFuncation(List<string> u_i_ratings, double[][] Q, double[][] P, double[] Bu, double[] Bi, double[][] Eij, double[][] Lui, double[][] Suv, List<string> users_id, List<string> items_id, List<List<int>> Friends_id, double[][] U_I_R)
        {
            double error = 0;
            double[] u_i_r = new double[u_i_ratings.Count];//用户实际评分
            double[] u_i_rp = new double[u_i_ratings.Count];//预测评分值

            #region//rating predict
            Parallel.For(0, u_i_ratings.Count, i =>
            //for (int i = 0; i < u_i_ratings.Count; i++)
            {
                string[] u_i_ids = u_i_ratings[i].Split(';');
                //u_i_r[i] = Convert.ToDouble(u_i_ids[2]);//1 / (1 + 1 / (Convert.ToDouble(u_i_ids[2])))
                u_i_r[i] = 1 / (1 + 1 / (Convert.ToDouble(u_i_ids[2])));
                u_i_rp[i] = rm + MatrixMulti(Q[users_id.IndexOf(u_i_ids[0])], P[items_id.IndexOf(u_i_ids[1])]) + Bu[users_id.IndexOf(u_i_ids[0])] + Bi[items_id.IndexOf(u_i_ids[1])];
            });
            #endregion
            double sum1 = 0;//预测误差
            double sum2 = 0;//基于好友关系的权重
            double sum3 = 0;//特征向量范数
            double sum3_P = 0;//特征向量范数P
            double sum3_Q = 0;//特征向量范数Q
            double sum4 = 0;//基于用户间兴趣相似度的权重
            double sum5 = 0;//基于用user-item topic 相似度的权重
            double sum6 = 0;//Lui
            double sum7 = 0;//Luu
            double sum8 = 0;//Lii

            #region//sum1
            Parallel.For(0, u_i_r.Length, i =>//======================================================================================
            //for (int i = 0; i < u_i_r.Length; i++)
            {
                lock (syn)
                {
                    sum1 = sum1 + 0.5 * (u_i_r[i] - u_i_rp[i]) * (u_i_r[i] - u_i_rp[i]);
                }
            });
            #endregion

            #region//sum3

            for (int i = 0; i < P.Length; i++)
            {
                Parallel.For(0, P[i].Length, j =>
                //for (int j = 0; j < P[i].Length; j++)
                {
                    lock (syn)
                    {
                        sum3_P = sum3_P + P[i][j] * P[i][j];
                    }
                });
            }
            for (int i = 0; i < Q.Length; i++)
            {
                Parallel.For(0, Q[i].Length, j =>
                //for (int j = 0; j < Q[i].Length; j++)
                {
                    lock (syn)
                    {
                        sum3_Q = sum3_Q + Q[i][j] * Q[i][j];
                    }
                });
            }
            //lanwta_Q = lanwta_P * sum3_P / sum3_Q;
            sum3 = 0.5 * (lanwta_P * sum3_P + lanwta_Q * sum3_Q);

            #endregion
            #region//sum2

            for (int i = 0; i < users_id.Count; i++)
            {
                sum2 = sum2 + Bu[i] * Bu[i];
            }
            for (int i = 0; i < items_id.Count; i++)
            {
                sum2 = sum2 + Bi[i] * Bi[i];
            }
            //beta = lanwta_P * sum3_P / sum2;
            sum2 = 0.5 * beta * sum2;

            #endregion
            #region//sum4

            for (int i = 0; i < items_id.Count; i++)
            {
                double[] Qv = new double[K];
                for (int l = 0; l < K; l++)
                {
                    for (int j = 0; j < Eij[i].Length; j++)
                    {
                        Qv[l] = Qv[l] + Eij[i][j] * P[j][l];
                    }
                }
                for (int l = 0; l < K; l++)
                    Qv[l] = P[i][l] - Qv[l];
                sum4 = sum4 + MatrixMulti(Qv, Qv);
            }
            //garma = lanwta_P * sum3_P / sum4;
            sum4 = 0.5 * garma * sum4;

            #endregion


            #region//sum5
            /*
            for (int i = 0; i < u_i_ratings.Count; i++)
            {
                string[] u_i_ids = u_i_ratings[i].Split(';');
                u_i_r[i] = Convert.ToDouble(u_i_ids[3]);

                sum5 = sum5 + Nuc[users_id.IndexOf(u_i_ids[0])] * (QPS[users_id.IndexOf(u_i_ids[0])][items_id.IndexOf(u_i_ids[1])] + rm - u_i_rp[i]) * (QPS[users_id.IndexOf(u_i_ids[0])][items_id.IndexOf(u_i_ids[1])] + rm - u_i_rp[i]);

                //sum5 *= Nuc[users_id.IndexOf(u_i_ids[0])];
            }
            sum5 = 0.5 * yita * sum5;
            */
            #endregion

            #region//sum6//////////////Lui

            double[] sum = new double[U_I_R.Length];
            List<double> a = new List<double>();
            for (int i = 0; i < Lui.Length; i++)
            {
                a = new List<double>();
                for (int j = 0; j < Lui[i].Length; j++)
                {
                    if (U_I_R[i][j] != 0)
                    {
                        a.Add(Lui[i][j]);
                    }
                }
                sum[i] = a.Sum();
            }

            Parallel.For(0, u_i_ratings.Count, i =>
            //for (int i = 0; i < u_i_ratings.Count; i++)
            {
                string[] u_i_ids = u_i_ratings[i].Split(';');

                //u_i_r[i] = Convert.ToDouble(u_i_ids[2]);//1 / (1 + 1 / (Convert.ToDouble(u_i_ids[2])))
                u_i_r[i] = 1 / (1 + 1 / (Convert.ToDouble(u_i_ids[2])));
                lock (syn)
                {
                    sum6 = sum6 + (Lui[users_id.IndexOf(u_i_ids[0])][items_id.IndexOf(u_i_ids[1])] / sum[users_id.IndexOf(u_i_ids[0])] + rm + Bu[users_id.IndexOf(u_i_ids[0])] + Bi[items_id.IndexOf(u_i_ids[1])] - u_i_rp[i]) * (Lui[users_id.IndexOf(u_i_ids[0])][items_id.IndexOf(u_i_ids[1])] / sum[users_id.IndexOf(u_i_ids[0])] + rm + Bu[users_id.IndexOf(u_i_ids[0])] + Bi[items_id.IndexOf(u_i_ids[1])] - u_i_rp[i]);
                }
            });
            //lui = 2 * lanwta_P * sum3_P / sum6;
            sum6 = 0.5 * lui * sum6;

            #endregion

            #region//sum7////////////Suv
            /*
            for (int i = 0; i < users_id.Count; i++)
            {
                double[] Qv = new double[K];
                for (int l = 0; l < K; l++)
                {
                    for (int j = 0; j < Suv[i].Length; j++)
                    {
                        //Qv[l] = Qv[l] + Suv[i][j] * Q[users_id.IndexOf(v_in_c[j])][l];
                        Qv[l] = Qv[l] + Suv[i][j] * Q[Friends_id[i][j]][l];
                    }
                }
                for (int l = 0; l < K; l++)
                {
                    Qv[l] = Q[i][l] - Qv[l];

                }
                sum7 = sum7 + MatrixMulti(Qv, Qv);
            }
            sum7 = 0.5 * suv * sum7;
            */
            #endregion


            FileStream fsw = new FileStream("error.txt", FileMode.Create);
            StreamWriter sw = new StreamWriter(fsw);
            sw.WriteLine("sum1: " + sum1);
            sw.WriteLine("sum2: " + sum2);
            sw.WriteLine("sum3: " + sum3);
            sw.WriteLine("sum4: " + sum4);
            sw.WriteLine("sum5: " + sum5);
            sw.WriteLine("sum6: " + sum6);
            sw.WriteLine("sum7: " + sum7);
            sw.Close();
            fsw.Close();
            error = sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
            return error;
        }

        public static void Nuc_QPS_Statistic(List<string> u_i_ratings, List<string> users_id, List<string> items_id, double[] Nuc, double[][] QPS)
        {
            #region//所有类别
            FileStream fs = new FileStream(yelp_test_path + "all_sub_c.txt", FileMode.Open);
            //FileStream fs = new FileStream(yelp_test_path + "allcate.txt", FileMode.Open);
            StreamReader sr = new StreamReader(fs);
            string[] rest_cates = Regex.Split(sr.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
            sr.Close();
            fs.Close();
            #endregion

            double sum = 0;
            double max = 0;
            for (int i = 0; i < users_id.Count; i++)
            {
                Nuc[i] = u_i_ratings.Count(s => s.Contains(users_id[i]));
                sum += Nuc[i];
                if (Nuc[i] > max)
                    max = Nuc[i];
            }
            for (int i = 0; i < users_id.Count; i++)
                Nuc[i] /= max;
            for (int i = 0; i < u_i_ratings.Count; i++)
            {
                string[] u_i_ids = u_i_ratings[i].Split(';');
                #region//读入用户兴趣和item主题向量
                double[] ui = new double[rest_cates.Length];
                #region//读入用户兴趣向量，用户评论聚类时已完成
                string ui_file = yelp_test_path + @"ui\" + u_i_ids[0] + ".txt";
                //string ui_file = yelp_test_path + "ui_allcate\\" + u_i_ids[0] + ".txt";
                FileStream fs1 = new FileStream(ui_file, FileMode.Open);
                StreamReader sr1 = new StreamReader(fs1);
                string[] str_line = Regex.Split(sr1.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
                sr1.Close();
                fs1.Close();
                double sum_ui = 0;
                for (int j = 0; j < rest_cates.Length; j++)
                {
                    ui[j] = Convert.ToDouble(str_line[j]);
                    sum_ui += ui[j];
                }
                if (sum_ui != 0)
                {
                    for (int j = 0; j < rest_cates.Length; j++)
                        ui[j] /= sum_ui;
                }
                #endregion
                double[] iiv = new double[rest_cates.Length];
                #region//读入item主题向量，用户评论聚类时已完成
                string iiv_file = yelp_test_path + "iiv\\" + u_i_ids[1] + ".txt";
                //string iiv_file = yelp_test_path + "iiv_allcate\\" + u_i_ids[1] + ".txt";
                FileStream fs2 = new FileStream(iiv_file, FileMode.Open);
                StreamReader sr2 = new StreamReader(fs2);
                string[] str_line2 = Regex.Split(sr2.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
                sr2.Close();
                fs2.Close();
                double sum_iiv = 0;
                for (int j = 0; j < rest_cates.Length; j++)
                {
                    iiv[j] = Convert.ToDouble(str_line2[j]);
                    sum_iiv += iiv[j];
                }
                if (sum_iiv != 0)
                {
                    for (int j = 0; j < rest_cates.Length; j++)
                        iiv[j] /= sum_iiv;
                }
                #endregion
                #endregion
                QPS[users_id.IndexOf(u_i_ids[0])][items_id.IndexOf(u_i_ids[1])] = Cos_Simi(ui, iiv);
            }
        }

        public static void Lui_Statistic_old(List<string> u_i_ratings, List<string> users_id, List<string> items_id, double[][] Lui)
        {
            double[] items_lat = new double[items_id.Count];//纬度
            double[] items_lon = new double[items_id.Count];//经度
            #region//读入所有item的gps
            FileStream fs1 = new FileStream(yelp_test_path + "pois.txt", FileMode.Open);
            StreamReader sr = new StreamReader(fs1);
            string[] items = Regex.Split(sr.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
            sr.Close();
            fs1.Close();
            for (int i = 0; i < items.Length - 1; i++)
            {
                items_lat[i] = Convert.ToDouble(items[i].Split(';')[1]);
                items_lon[i] = Convert.ToDouble(items[i].Split(';')[2]);
            }

            #endregion

            double[] user_lat = new double[users_id.Count];
            double[] user_lon = new double[users_id.Count];
            #region//读入所有user的gps
            fs1 = new FileStream(yelp_test_path + "users.txt", FileMode.Open);
            sr = new StreamReader(fs1);
            string[] users = Regex.Split(sr.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
            sr.Close();
            fs1.Close();
            for (int i = 0; i < users.Length - 1; i++)
            {
                user_lat[i] = Convert.ToDouble(users[i].Split(';')[1]);
                user_lon[i] = Convert.ToDouble(users[i].Split(';')[2]);
            }
            #endregion
            //double value = 0;
            Parallel.For(0, u_i_ratings.Count, i =>
            // for (int i = 0; i < u_i_ratings.Count; i++)
            {
                string[] u_i_ids = u_i_ratings[i].Split(';');
                int user_id_index = users_id.IndexOf(u_i_ids[0]);
                int item_id_index = items_id.IndexOf(u_i_ids[1]);
                double value = Convert.ToDouble(u_i_ratings[i].Split(';')[u_i_ratings[i].Split(';').Length - 1]);// GetDistance(user_lat[user_id_index], user_lon[user_id_index], items_lat[item_id_index], items_lon[item_id_index]);
                Lui[user_id_index][item_id_index] = Math.Pow(Math.E, -value);
            });

            #region//归一化
            List<double> ma = new List<double>();
            List<double> mi = new List<double>();
            for (int i = 0; i < users_id.Count; i++)
            {
                List<double> temp = new List<double>();
                Parallel.For(0, items_id.Count, j =>
                //for (int j = 0; j < items_id.Count; j++)
                {
                    if (Lui[i][j] != 0)
                    {
                        lock (syn)
                        {
                            temp.Add(Lui[i][j]);
                        }
                    }
                });
                if (temp.Count > 0)
                {
                    ma.Add(temp.Max());
                    mi.Add(temp.Min());
                }
                else
                {
                    ma.Add(rm);
                    mi.Add(rm);
                }
            }
            double max = ma.Max() + 0.0000001;
            double min = mi.Min() - 0.0000001;

            //for (int i = 0; i < users_id.Count; i++)
            //{
            //    double max = Lui[i].Max();
            //    double min = Lui[i].Min();
            //    for (int j = 0; j < items_id.Count; j++)
            //    {
            //        Lui[i][j] = (Lui[i][j] - min) / (max - min);
            //    }
            //}
            Parallel.For(0, u_i_ratings.Count, i =>
            //for (int i = 0; i < u_i_ratings.Count; i++)
            {
                string[] u_i_ids = u_i_ratings[i].Split(';');
                int user_id_index = users_id.IndexOf(u_i_ids[0]);
                int item_id_index = items_id.IndexOf(u_i_ids[1]);

                Lui[user_id_index][item_id_index] = (Lui[user_id_index][item_id_index] - min) / (max - min);
            });

            #endregion
        }

        public static void Lui_Statistic(List<string> u_i_ratings, List<string> users_id, List<string> items_id, double[][] Lui)
        {
            double[] items_lat = new double[items_id.Count];//纬度
            double[] items_lon = new double[items_id.Count];//经度
            #region//读入所有item的gps
            FileStream fs1 = new FileStream(yelp_test_path + "pois.txt", FileMode.Open);
            StreamReader sr = new StreamReader(fs1);
            string[] items = Regex.Split(sr.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
            sr.Close();
            fs1.Close();
            for (int i = 0; i < items.Length - 1; i++)
            {
                items_lat[i] = Convert.ToDouble(items[i].Split(';')[1]);
                items_lon[i] = Convert.ToDouble(items[i].Split(';')[2]);
            }

            #endregion

            //double[] user_lat = new double[users_id.Count];
            //double[] user_lon = new double[users_id.Count];
            List<List<double>> user_lat = new List<List<double>>();
            List<List<double>> user_lon = new List<List<double>>();
            for (int i = 0; i < users_id.Count; i++)
            {
                user_lat.Add(new List<double>());
                user_lon.Add(new List<double>());
            }
            #region//读入所有user的gps
            fs1 = new FileStream(yelp_test_path + "user_centers.txt", FileMode.Open);
            sr = new StreamReader(fs1);
            string[] users = Regex.Split(sr.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
            sr.Close();
            fs1.Close();
            int id = 0;
            for (int i = 0; i < users.Length - 1; i++)
            {
                id = users_id.IndexOf(users[i].Split(';')[0]);
                for (int j = 1; j < users[i].Split(';').Length - 1; j++)
                {
                    user_lat[id].Add(Convert.ToDouble(users[i].Split(';')[j].Split(' ')[0]));
                    user_lon[id].Add(Convert.ToDouble(users[i].Split(';')[j].Split(' ')[1]));
                }
            }
            fs1 = new FileStream(yelp_test_path + "users.txt", FileMode.Open);
            sr = new StreamReader(fs1);
            users = Regex.Split(sr.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
            sr.Close();
            fs1.Close();
            for (int i = 0; i < users.Length - 1; i++)
            {
                user_lat[i].Add(Convert.ToDouble(users[i].Split(';')[1]));
                user_lon[i].Add(Convert.ToDouble(users[i].Split(';')[2]));
            }
            #endregion
            //double value = 0;
            Parallel.For(0, u_i_ratings.Count, i =>
            // for (int i = 0; i < u_i_ratings.Count; i++)
            {
                string[] u_i_ids = u_i_ratings[i].Split(';');
                int user_id_index = users_id.IndexOf(u_i_ids[0]);
                int item_id_index = items_id.IndexOf(u_i_ids[1]);
                //double value = Convert.ToDouble(u_i_ratings[i].Split(';')[u_i_ratings[i].Split(';').Length - 1]);// GetDistance(user_lat[user_id_index], user_lon[user_id_index], items_lat[item_id_index], items_lon[item_id_index]);
                double value = 1000000;
                double distance = 0;// GetDistance(user_lat[user_id_index], user_lon[user_id_index], items_lat[item_id_index], items_lon[item_id_index]);
                for (int j = 0; j < user_lat[user_id_index].Count; j++)
                {
                    distance = GetDistance(user_lat[user_id_index][j], user_lon[user_id_index][j], items_lat[item_id_index], items_lon[item_id_index]);
                    if (distance < value)
                    {
                        value = distance;
                    }
                }
                Lui[user_id_index][item_id_index] = Math.Pow(Math.E, -value);
            });

            #region//归一化
            List<double> ma = new List<double>();
            List<double> mi = new List<double>();
            for (int i = 0; i < users_id.Count; i++)
            {
                List<double> temp = new List<double>();
                Parallel.For(0, items_id.Count, j =>
                //for (int j = 0; j < items_id.Count; j++)
                {
                    if (Lui[i][j] != 0)
                    {
                        lock (syn)
                        {
                            temp.Add(Lui[i][j]);
                        }
                    }
                });
                if (temp.Count > 0)
                {
                    ma.Add(temp.Max());
                    mi.Add(temp.Min());
                }
                else
                {
                    ma.Add(rm);
                    mi.Add(rm);
                }
            }
            double max = ma.Max() + 0.0000001;
            double min = mi.Min() - 0.0000001;

            //for (int i = 0; i < users_id.Count; i++)
            //{
            //    double max = Lui[i].Max();
            //    double min = Lui[i].Min();
            //    for (int j = 0; j < items_id.Count; j++)
            //    {
            //        Lui[i][j] = (Lui[i][j] - min) / (max - min);
            //    }
            //}
            Parallel.For(0, u_i_ratings.Count, i =>
            //for (int i = 0; i < u_i_ratings.Count; i++)
            {
                string[] u_i_ids = u_i_ratings[i].Split(';');
                int user_id_index = users_id.IndexOf(u_i_ids[0]);
                int item_id_index = items_id.IndexOf(u_i_ids[1]);

                Lui[user_id_index][item_id_index] = (Lui[user_id_index][item_id_index] - min) / (max - min);
            });

            #endregion
        }

        public static void Luu_Statistic(List<string> u_i_ratings, List<string> users_id, int[][] FriendsCircle, double[][] Luu)
        {


            double[] user_lat = new double[users_id.Count];
            double[] user_lon = new double[users_id.Count];
            #region//读入所有user的gps

            FileStream fs1 = new FileStream(yelp_test_path + "new_data.txt", FileMode.Open);
            StreamReader sr1 = new StreamReader(fs1);
            string[] content = Regex.Split(sr1.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
            sr1.Close();
            fs1.Close();

            for (int i = 0; i < content.Length - 1; i++)
            {
                user_lat[users_id.IndexOf(content[i].Split(';')[0])] = Math.Round(Convert.ToDouble(content[i].Split(';')[4].Split(',')[0].Split('(')[1]) * 10000) / 10000;
                user_lon[users_id.IndexOf(content[i].Split(';')[0])] = Math.Round(Convert.ToDouble(content[i].Split(';')[4].Split(',')[1].Split(')')[0]) * 10000) / 10000;
            }
            #endregion

            double value = 0;
            for (int i = 0; i < users_id.Count; i++)
            {
                Luu[i] = new double[FriendsCircle[i].Length];
                for (int j = 0; j < FriendsCircle[i].Length; j++)
                {
                    value = GetDistance(user_lat[i], user_lon[i], user_lat[FriendsCircle[i][j]], user_lon[FriendsCircle[i][j]]);
                    //Luu[i][j] = Fourier1_Luu(value);// Gauss1_Luu(value);
                    //if (CF_Luu == "RestaurantsGauss2")
                    //{
                    //    Luu[i][j] = CF_Luu_RestaurantsGauss2(value);
                    //}
                    //if (CF_Luu == "NightlifeGauss2")
                    //{
                    //    Luu[i][j] = CF_Luu_NightlifeGauss2(value);
                    //}
                    //if (CF_Luu == "Arts & EntertainmentGauss2")
                    //{
                    //    Luu[i][j] = CF_Luu_ArtsandEntertainmentGauss2(value);
                    //}
                    //if (CF_Luu == "RestaurantsPoly4")
                    //{
                    //    Luu[i][j] = CF_Luu_RestaurantsPoly4(value);
                    //}
                    //if (CF_Luu == "RestaurantsPoly5")
                    //{
                    //    Luu[i][j] = CF_Luu_RestaurantsPoly5(value);
                    //}
                    //if (CF_Luu == "RestaurantsPoly6")
                    //{
                    //    Luu[i][j] = CF_Luu_RestaurantsPoly6(value);
                    //}
                    //if (CF_Luu == "RestaurantsSin2")
                    //{
                    //    Luu[i][j] = CF_Luu_RestaurantsSin2(value);
                    //}
                }
            }
            for (int i = 0; i < users_id.Count; i++)
            {
                if (Luu[i].Length > 0)
                {
                    double max = Luu[i].Max() + 0.0000001;
                    double min = Luu[i].Min() - 0.0000001;

                    if (max == min)
                    {
                        for (int j = 0; j < FriendsCircle[i].Length; j++)
                        {
                            Luu[i][j] = (double)1 / (double)FriendsCircle[i].Length;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < FriendsCircle[i].Length; j++)
                        {
                            Luu[i][j] = (max - Luu[i][j]) / (max - min);
                        }
                        for (int j = 0; j < FriendsCircle[i].Length; j++)
                        {
                            Luu[i][j] = Luu[i][j] / Luu[i].Sum();
                        }
                    }
                }
            }
        }


        public static void Lii_Statistic(List<string> u_i_ratings, List<string> items_id, double[][] Lii)
        {


            double[] items_lat = new double[items_id.Count];//纬度
            double[] items_lon = new double[items_id.Count];//经度
            #region//读入所有item的gps
            for (int i = 0; i < items_id.Count; i++)
            {
                string item_gps_file = yelp_test_path + "gps_items\\" + items_id[i] + ".txt";
                FileStream fs1 = new FileStream(item_gps_file, FileMode.Open);
                StreamReader sr1 = new StreamReader(fs1);
                string str_line = sr1.ReadLine();
                while (str_line != null)
                {
                    if (str_line.Contains("latitude\":"))
                    {
                        string[] str_lat = str_line.Split(' ');
                        items_lat[i] = Math.Round(Convert.ToDouble(str_lat[str_lat.Length - 1]) * 10000) / 10000;
                    }
                    if (str_line.Contains("longitude\":"))
                    {
                        string[] str_lat = str_line.Split(' ');
                        items_lon[i] = Math.Round(Convert.ToDouble(str_lat[str_lat.Length - 1].Substring(0, str_lat[str_lat.Length - 1].Length - 1)) * 10000) / 10000;
                    }
                    str_line = sr1.ReadLine();
                }
                sr1.Close();
                fs1.Close();
            }
            #endregion
            double value = 0;
            for (int i = 0; i < items_id.Count; i++)
            {
                for (int j = 0; j < items_id.Count; j++)
                {
                    value = GetDistance(items_lat[i], items_lon[i], items_lat[j], items_lon[j]);
                    if (value < Math.E)
                    {
                        Lii[i][j] = 1;
                    }
                    else
                    {
                        Lii[i][j] = (double)1 / Math.Log(value);
                    }
                }
            }


            //for (int i = 0; i < items_id.Count; i++)
            //{
            //    for (int j = 0; j < items_id.Count; j++)
            //    {
            //        value = GetDistance(items_lat[i], items_lon[i], items_lat[j], items_lon[j]);
            //        Lii[i][j] = Gauss2_Lui(value);
            //    }
            //}
            //#region//归一化
            //List<double> ma = new List<double>();
            //List<double> mi = new List<double>();
            //for (int i = 0; i < items_id.Count; i++)
            //{
            //    ma.Add(Lii[i].Max());
            //    mi.Add(Lii[i].Min());
            //}
            //double max = ma.Max() + 0.0000001;
            //double min = mi.Min() - 0.0000001;

            //for (int i = 0; i < items_id.Count; i++)
            //{
            //    for (int j = 0; j < items_id.Count; j++)
            //    {
            //        Lii[i][j] = (Lii[i][j] - min) / (max - min);
            //    }
            //}

            //#endregion

        }

        public static void SimiUI_new(List<string> u_i_ratings, List<string> items_id, double[][] Eij)//CirCon2a(仅计算truster在该类内的评论数)///////////////////////////////////////////////////////////////////////////////////////////////
        {
            FileStream fs1 = new FileStream(yelp_test_path + "pois.txt", FileMode.Open);
            StreamReader sr = new StreamReader(fs1);
            string[] items = Regex.Split(sr.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
            sr.Close();
            fs1.Close();
            List<double> emotions = new List<double>();
            for (int i = 0; i < items.Length - 1; i++)
            {
                emotions.Add(Convert.ToDouble(items[i].Split(';')[3]));
            }
            for (int i = 0; i < items_id.Count; i++)
            {
                #region

                Eij[i] = new double[items_id.Count];

                for (int j = 0; j < Eij[i].Length; j++)
                {
                    //Eij[i][j] = (double)1 / Math.Abs(emotions[i] - emotions[j]);
                    Eij[i][j] = Math.Pow(Math.E, (-Math.Abs(emotions[i] - emotions[j])));
                }

                #region//归一化
                double sum = 0;
                for (int j = 0; j < Eij[i].Length; j++)
                    sum += Eij[i][j];
                if (sum == 0)
                    continue;
                for (int j = 0; j < Eij[i].Length; j++)
                    Eij[i][j] /= sum;
                #endregion
                //=====================================================================================
                //sum = 0;
                //for (int j = 0; j < Iuv[i].Length; j++)
                //{
                //    Iuv[i][j] = Iuv[i][j] * Luu[i][j];
                //    sum += Iuv[i][j];
                //}
                //if (sum == 0)
                //    continue;
                //for (int j = 0; j < Iuv[i].Length; j++)
                //    Iuv[i][j] = Iuv[i][j]/sum;
                //====================================================================================
                #endregion
                //FileStream fss = new FileStream(@"restaurants\result_n+Iuv\SUV_entropy_same cat\归一化\" + i + ".txt", FileMode.Open);
                //StreamReader srr = new StreamReader(fss);
                //string[] stt = Regex.Split(srr.ReadToEnd(), "\r\n", RegexOptions.IgnoreCase);
                //Iuv[i]=new double[stt.Length - 1];
                //for (int j = 0; j < stt.Length - 1; j++)
                //{
                //    Iuv[i][j] = Convert.ToDouble(stt[j]);
                //}

            }
        }

        public static double MatrixMulti(double[] M, double[] N)
        {
            double res = 0;
            for (int i = 0; i < M.Length; i++)
            {
                res = res + M[i] * N[i];
            }
            return res;
        }

        public static double NormalDistribution(Random rand)
        {
            double u1 = 0;
            double u2 = 0;
            double v1 = 0;
            double v2 = 0;
            double s = 0;
            double z1 = 0;
            double z2 = 0;
            while (s > 1 || s == 0)
            {
                u1 = rand.NextDouble();
                u2 = rand.NextDouble();
                v1 = 2 * u1 - 1;
                v2 = 2 * u2 - 1;
                s = v1 * v1 + v2 * v2;
            }
            z1 = Math.Sqrt(-2 * Math.Log(s) / s) * v1;
            z2 = Math.Sqrt(-2 * Math.Log(s) / s) * v2;

            return z1; //返回两个服从正态分布N(0,1)的随机数z0 和 z1
        }

        public static double Cos_Simi(double[] tv1, double[] tv2)
        {
            double topic_simi = 0;
            double a_b = 0;
            double a_a = 0;
            double b_b = 0;
            for (int i = 0; i < tv1.Length; i++)
            {
                a_b += tv1[i] * tv2[i];
                a_a += tv1[i] * tv1[i];
                b_b += tv2[i] * tv2[i];
            }
            if (a_a == 0 || b_b == 0)
                topic_simi = 0;
            else
                topic_simi = a_b / (Math.Sqrt(a_a) * Math.Sqrt(b_b));
            return topic_simi;
        }

        public static double GetDistance(double lat1, double lng1, double lat2, double lng2)
        {
            double radLat1 = rad(lat1);
            double radLat2 = rad(lat2);
            double a = radLat1 - radLat2;
            double b = rad(lng1) - rad(lng2);

            double s = 2 * Math.Asin(Math.Sqrt(Math.Pow(Math.Sin(a / 2), 2) +
             Math.Cos(radLat1) * Math.Cos(radLat2) * Math.Pow(Math.Sin(b / 2), 2)));
            s = s * EARTH_RADIUS * 1000;
            s = Math.Round(s * 10000) / 10000;
            return s;
        }

        public static double rad(double d)
        {
            return d * Math.PI / 180.0;
        }

        public static void Precision_Recall(List<string> u_i_rating_test, List<string> users_id, List<string> items_id, double[][] U_I_R, double[][] Q, double[][] P, double[] Bu, double[] Bi, double[] u_i_rp_t, double[] p, double[] r, double[] MAP)
        {
            //p = new double[100];
            //r = new double[100];
            int[][] count = new int[p.Length][];
            for (int i = 0; i < count.Length; i++)
            {
                count[i] = new int[users_id.Count];
            }
            //double[][] U_I_R_new=new double[users_id.Count][];
            List<List<double>> U_I_R_new = new List<List<double>>();
            List<List<double>> U_I_R_test = new List<List<double>>();
            List<List<double>> U_I_R_temp = new List<List<double>>();
            for (int i = 0; i < users_id.Count; i++)
            {
                //U_I_R_new[i] = new double[items_id.Count];
                U_I_R_new.Add(new List<double>());
                U_I_R_test.Add(new List<double>());
                U_I_R_temp.Add(new List<double>());
                for (int j = 0; j < items_id.Count; j++)
                {
                    if (U_I_R[i][j] == 0)
                    {
                        U_I_R_new[i].Add(rm + MatrixMulti(Q[i], P[j]) + Bu[i] + Bi[j]);
                        U_I_R_test[i].Add(0);
                        U_I_R_temp[i].Add(rm + MatrixMulti(Q[i], P[j]) + Bu[i] + Bi[j]);
                    }
                    else
                    {
                        U_I_R_new[i].Add(0);//用户去过
                        U_I_R_temp[i].Add(0);
                        U_I_R_test[i].Add(0);
                    }
                }
            }
            for (int i = 0; i < u_i_rating_test.Count; i++)//读入测试数据
            {
                string[] u_i_ids = u_i_rating_test[i].Split(';');
                string user_id = u_i_ids[0];
                string item_id = u_i_ids[1];
                //U_I_R_test[users_id.IndexOf(user_id)][items_id.IndexOf(item_id)] = Convert.ToDouble(u_i_ids[2]);//1 / (1 + 1 / (Convert.ToDouble(u_i_ids[2])));////////
                U_I_R_test[users_id.IndexOf(user_id)][items_id.IndexOf(item_id)] = 1 / (1 + 1 / (Convert.ToDouble(u_i_ids[2])));////////

            }
            for (int i = 0; i < p.Length; i++)
            {
                //count[i] = 0;
                double max = 0;
                for (int j = 0; j < users_id.Count; j++)
                {
                    max = U_I_R_temp[j].Max();
                    U_I_R_temp[j].RemoveAt(U_I_R_temp[j].IndexOf(max));
                    if (max != 0)
                    {
                        //if (U_I_R_new[j][U_I_R_new[j].IndexOf(max)] != 0 && U_I_R_test[j][U_I_R_new[j].IndexOf(max)] != 0)//top1 是用户训练集中没去过的地方 && 测试集中去过  （即正确样本）
                        //{
                        //    count[i]++;
                        //}
                        if (U_I_R_test[j][U_I_R_new[j].IndexOf(max)] != 0)//top1 是用户训练集中没去过的地方 && 测试集中去过  （即正确样本）
                        {
                            count[i][j]++;
                        }
                    }
                    else//top1 是用户训练集中去过的地方 应该不推荐 pass
                    {

                    }
                }
            }
            //计算实际测试集中正确的样本个数
            int N = 0;
            int[] N_r = new int[U_I_R_test.Count];
            for (int i = 0; i < U_I_R_test.Count; i++)
            {
                for (int j = 0; j < U_I_R_test[i].Count; j++)
                {
                    if (U_I_R_test[i][j] != 0)
                    {
                        N++;
                        N_r[i]++;
                    }
                }
            }
            //计算Precision
            int n = 0;
            for (int i = 0; i < p.Length; i++)
            {
                n = n + count[i].Sum();
                p[i] = (double)n / (double)((i + 1) * users_id.Count);
            }
            //计算Recall
            n = 0;
            List<double> Recall = new List<double>();
            for (int i = 0; i < p.Length; i++)
            {
                Recall = new List<double>();
                //n = n + count[i].Sum();
                for (int j = 0; j < count[i].Length; j++)
                {
                    if (N_r[j] != 0)
                    {
                        n = 0;
                        for (int k = 0; k < i + 1; k++)
                        {
                            n = n + count[k][j];
                        }
                        Recall.Add((double)n / (double)N_r[j]);
                    }
                }
                r[i] = Recall.Average();
            }

            // 计算MAP
            n = 0;
            List<double> MeanPrecision = new List<double>();
            for (int j = 0; j < users_id.Count; j++)
            {
                if (N_r[j] != 0)
                {
                    List<double> Precision = new List<double>();
                    for (int i = 0; i < p.Length; i++)
                    {
                        n = 0;
                        for (int k = 0; k < i + 1; k++)
                        {
                            n = n + count[k][j];
                        }
                        Precision.Add((double)n / (double)(i + 1));
                    }
                    MeanPrecision.Add(Precision.Sum() / (double)N_r[j]);
                }
            }

            MAP[0] = MeanPrecision.Average();
            Console.WriteLine("MAP:" + MAP[0]);
        }



        public static double[][] users_similarity_normalized_new(List<double> items_average_rating, List<string> user_id, List<string> item_id, double[][] ratings, List<List<int>> Friends_id, double[][] similarity)
        {
            List<List<double>> rating = new List<List<double>>();
            for (int i = 0; i < item_id.Count; i++)
            {
                rating.Add(new List<double>());
                for (int j = 0; j < user_id.Count; j++)
                {
                    if (ratings[j][i] != 0)
                    {
                        rating[i].Add(ratings[j][i]);
                    }
                }
                if (rating[i].Count == 0)
                {
                    rating[i].Add(rm);
                }
                items_average_rating.Add(rating[i].Average());
            }
            similarity = new double[user_id.Count][];
            List<List<double>> sim = new List<List<double>>();
            List<List<int>> index = new List<List<int>>();
            for (int i = 0; i < user_id.Count; i++)
            {
                similarity[i] = new double[Friends_id[i].Count];
                sim.Add(new List<double>());
                index.Add(new List<int>());
            }

            // int n = 0;
            //Parallel.For(0, item_id.Count, i =>
            for (int i = 0; i < user_id.Count; i++)
            {
                //similarity[i] = new double[item_id.Count];

                Parallel.For(0, Friends_id[i].Count, j =>
                //for (int j = i; j < item_id.Count; j++)
                {

                    float numerator = 0;
                    float denominator = 0;
                    float denominator1 = 0;
                    float denominator2 = 0;
                    bool flag = false;

                    for (int k = 0; k < item_id.Count; k++)
                    {
                        if (ratings[i][k] != 0 && ratings[Friends_id[i][j]][k] != 0)
                        {
                            numerator = numerator + (float)(ratings[i][k] - items_average_rating[k] - 0.0000001) * (float)(ratings[Friends_id[i][j]][k] - items_average_rating[k] - 0.0000001);
                            denominator1 = denominator1 + (float)Math.Pow(ratings[Friends_id[i][j]][k] - items_average_rating[k] - 0.0000001, 2);
                            denominator2 = denominator2 + (float)Math.Pow(ratings[Friends_id[i][j]][k] - items_average_rating[k] - 0.0000001, 2);
                            flag = true;
                        }
                    }
                    if (flag)
                    {
                        denominator = (float)((float)Math.Sqrt(denominator1) * (float)Math.Sqrt(denominator2));

                        similarity[i][j] = numerator / denominator;

                    }
                });
                Console.WriteLine(i);
            }//);
            double min = (double)0;
            Parallel.For(0, user_id.Count, i =>
            //for (int i = 0; i < item_id.Count; i++)
            {
                for (int j = 0; j < Friends_id[i].Count; j++)
                {
                    lock (syn)
                    {
                        if (similarity[i][j] != 0 && similarity[i][j] < min)
                        {
                            min = similarity[i][j];
                        }
                    }
                }
            });

            min = min - (float)0.0000001;

            Parallel.For(0, user_id.Count, i =>
            {

                for (int j = 0; j < Friends_id[i].Count; j++)
                {
                    lock (syn)
                    {
                        if (similarity[i][j] != 0)
                        {
                            similarity[i][j] = similarity[i][j] - min;
                        }
                    }
                }

            });
            Parallel.For(0, user_id.Count, i =>
            //for (int i = 0; i < item_id.Count; i++)
            {
                double sum = (double)0.0000001;
                for (int j = 0; j < Friends_id[i].Count; j++)
                {
                    if (similarity[i][j] != 0)
                        sum = sum + similarity[i][j];
                }
                if (sum != 0.0000001)
                {
                    for (int j = 0; j < Friends_id[i].Count; j++)
                    {
                        if (similarity[i][j] != 0)
                            similarity[i][j] = (double)(similarity[i][j] / sum);
                    }
                }
                else
                {
                    for (int j = 0; j < Friends_id[i].Count; j++)
                    {
                        similarity[i][j] = (double)1 / (double)Friends_id[i].Count;
                    }
                }

            });
            if (!Directory.Exists(category + "\\user_based similarity_normalized_new\\"))
            {
                Directory.CreateDirectory(category + "\\user_based similarity_normalized_new\\");
            }

            for (int i = 0; i < user_id.Count; i++)
            {
                FileStream fsw = new FileStream(category + "\\user_based similarity_normalized_new\\" + i + ".txt", FileMode.Create);
                StreamWriter sw = new StreamWriter(fsw);
                for (int j = 0; j < Friends_id[i].Count; j++)
                {
                    sw.Write(similarity[i][j] + ";");
                }
                sw.Close();
                fsw.Close();
            }

            return similarity;
        }

    }
}
