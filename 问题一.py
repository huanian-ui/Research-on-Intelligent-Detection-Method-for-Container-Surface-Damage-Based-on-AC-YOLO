import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# 全局中文显示与风格
# =======================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False                       # 负号正常显示
sns.set_style("whitegrid")
# 设置中文字体和科研配色
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

COLORS = ['#90D8A6', '#83A1E7', '#E992A9', '#D2CAF8', '#F7AF7F', '#B0D9F9', '#E7B6BC', '#B0CDED']


def percent_to_float(x):
    """
    将类似 '92.13%' 或 '92.13' -> float(92.13)
    """
    if isinstance(x, str):
        x = x.strip().replace('%', '')
    try:
        return float(x)
    except:
        return np.nan


def safe_read_csv(path, **kwargs):
    if os.path.exists(path):
        print(f"[INFO] 读取 {path}")
        return pd.read_csv(path, **kwargs)
    else:
        print(f"[WARN] 未找到 {path} ，后续相关分析会跳过。")
        return None


def draw_per_class_metrics(per_class_df):
    """
    绘制每个类别的 精确率/召回率/F1 对比柱状图
    """
    if per_class_df is None or len(per_class_df) == 0:
        return

    melted = per_class_df.melt(
        id_vars=['类别'],
        value_vars=['精确率_float', '召回率_float', 'F1_float'],
        var_name='指标',
        value_name='数值'
    )

    # 替换指标中文显示
    melted['指标'] = melted['指标'].replace({
        '精确率_float': '精确率(Precision)',
        '召回率_float': '召回率(Recall)',
        'F1_float': 'F1分数(F1-score)'
    })

    plt.figure(figsize=(10,6))
    sns.barplot(
        x='类别', y='数值', hue='指标', data=melted,
        palette=COLORS[:3], edgecolor='black', alpha=0.8
    )
    plt.ylabel('数值 (%)')
    plt.ylim(0, 105)
    plt.title('按类别的精确率 / 召回率 / F1 分数', fontsize=14, fontweight='bold')
    plt.legend(title='指标', loc='upper right')
    for idx, row in melted.iterrows():
        plt.text(
            x=idx % len(per_class_df),
            y=row['数值'] + 1,
            s=f"{row['数值']:.1f}%",
            ha='center',
            fontsize=8,
            rotation=0
        )
    plt.tight_layout()
    plt.savefig('per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def draw_train_val_curves(history_df):
    """
    绘制训练/验证准确率与损失曲线，以及训练-验证差值随epoch的走势
    额外给出过拟合风险提示线
    """
    if history_df is None or len(history_df) == 0:
        return None

    # 曲线1：准确率
    plt.figure(figsize=(10,4))
    plt.plot(history_df['epoch'], history_df['train_acc'], marker='o', label='训练准确率', linewidth=2)
    plt.plot(history_df['epoch'], history_df['val_acc'], marker='s', label='验证准确率', linewidth=2)
    plt.xlabel('训练轮次 (epoch)')
    plt.ylabel('准确率 (%)')
    plt.title('训练/验证 准确率曲线', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('train_val_accuracy_curve_eval.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 曲线2：损失
    plt.figure(figsize=(10,4))
    plt.plot(history_df['epoch'], history_df['train_loss'], marker='o', label='训练损失', linewidth=2)
    plt.plot(history_df['epoch'], history_df['val_loss'], marker='s', label='验证损失', linewidth=2)
    plt.xlabel('训练轮次 (epoch)')
    plt.ylabel('损失 (loss)')
    plt.title('训练/验证 损失曲线', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('train_val_loss_curve_eval.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 曲线3：过拟合指标 (训练acc - 验证acc)
    gap = history_df['train_acc'] - history_df['val_acc']
    plt.figure(figsize=(10,4))
    plt.plot(history_df['epoch'], gap, marker='o', linewidth=2, color='#E992A9')
    plt.axhline(y=5, linestyle='--', color='gray', label='5个百分点警戒线')
    plt.xlabel('训练轮次 (epoch)')
    plt.ylabel('训练-验证 准确率差值 (百分点)')
    plt.title('训练-验证 准确率差值走势（过拟合风险）', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('train_val_gap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 计算一些量化信息供总结文本使用
    best_val_idx = history_df['val_acc'].idxmax()
    best_epoch_row = history_df.loc[best_val_idx]

    info = {
        "best_epoch": int(best_epoch_row['epoch']),
        "best_val_acc": float(best_epoch_row['val_acc']),
        "train_acc_at_best": float(best_epoch_row['train_acc']),
        "gap_at_best": float(best_epoch_row['train_acc'] - best_epoch_row['val_acc']),
        "final_epoch": int(history_df.iloc[-1]['epoch']),
        "final_val_acc": float(history_df.iloc[-1]['val_acc']),
        "final_train_acc": float(history_df.iloc[-1]['train_acc']),
        "final_gap": float(gap.iloc[-1]),
        "val_loss_trend_last3": float(history_df['val_loss'].diff().fillna(0).tail(3).mean())
    }

    return info


def draw_test_pred_distribution(pred_df):
    """
    分析测试集预测分布：哪个类别在预测结果里出现得最多
    这可以反映模型在真实评估场景下的偏好/输出倾向
    """
    if pred_df is None or len(pred_df) == 0:
        return None

    dist = pred_df['class_name'].value_counts().reset_index()
    dist.columns = ['类别', '数量']
    dist['占比(%)'] = dist['数量'] / dist['数量'].sum() * 100

    plt.figure(figsize=(8,5))
    sns.barplot(
        x='类别', y='数量',
        data=dist,
        palette=COLORS[:len(dist)],
        edgecolor='black', alpha=0.85
    )
    for i, row in dist.iterrows():
        plt.text(i, row['数量'] + 0.5, f"{row['占比(%)']:.1f}%", ha='center', fontsize=10)
    plt.title('测试集预测类别分布 (推测现场占比/输出偏向)', fontsize=14, fontweight='bold')
    plt.ylabel('数量')
    plt.xlabel('预测类别')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_pred_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    return dist


def compute_risk_flags(per_class_df):
    """
    风险提示：
    - 召回率低：漏检风险（特别是对安全/质量检查任务来说很严重）
    - 精确率低：误报风险（可能导致报警过多，增加人工成本）
    """
    if per_class_df is None or len(per_class_df)==0:
        return {"low_recall": [], "low_precision": [], "weakest_class": None}

    weakest_row = per_class_df.sort_values('F1_float').iloc[0]

    low_recall_classes = per_class_df[ per_class_df['召回率_float'] < 60 ]['类别'].tolist()
    low_precision_classes = per_class_df[ per_class_df['精确率_float'] < 60 ]['类别'].tolist()

    return {
        "weakest_class": weakest_row['类别'],
        "weakest_class_f1": float(weakest_row['F1_float']),
        "low_recall": low_recall_classes,
        "low_precision": low_precision_classes
    }


def build_dashboard_figure(per_class_df, history_df, pred_dist_df):
    """
    生成一个总览大盘图 dashboard_overview.png
    4宫格：
    1. 各类别F1
    2. 训练/验证准确率
    3. 训练/验证损失
    4. 测试集预测分布
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 子图1：F1按类别
    if per_class_df is not None and len(per_class_df) > 0:
        f1_sorted = per_class_df.sort_values('F1_float', ascending=False)
        axes[0,0].bar(
            f1_sorted['类别'],
            f1_sorted['F1_float'],
            color=COLORS[:len(f1_sorted)], edgecolor='black', alpha=0.8
        )
        for i, row in f1_sorted.iterrows():
            axes[0,0].text(
                x=list(f1_sorted['类别']).index(row['类别']),
                y=row['F1_float'] + 1,
                s=f"{row['F1_float']:.1f}%",
                ha='center', fontsize=9
            )
        axes[0,0].set_title('类别级别F1分数（越高越稳）', fontsize=13, fontweight='bold')
        axes[0,0].set_ylabel('F1分数 (%)')
        axes[0,0].grid(axis='y', alpha=0.3)

    # 子图2：准确率曲线
    if history_df is not None and len(history_df) > 0:
        axes[0,1].plot(history_df['epoch'], history_df['train_acc'],
                       marker='o', label='训练准确率', linewidth=2)
        axes[0,1].plot(history_df['epoch'], history_df['val_acc'],
                       marker='s', label='验证准确率', linewidth=2)
        axes[0,1].set_title('训练 / 验证 准确率走势', fontsize=13, fontweight='bold')
        axes[0,1].set_xlabel('epoch')
        axes[0,1].set_ylabel('准确率 (%)')
        axes[0,1].legend()
        axes[0,1].grid(alpha=0.3)

    # 子图3：损失曲线
    if history_df is not None and len(history_df) > 0:
        axes[1,0].plot(history_df['epoch'], history_df['train_loss'],
                       marker='o', label='训练损失', linewidth=2)
        axes[1,0].plot(history_df['epoch'], history_df['val_loss'],
                       marker='s', label='验证损失', linewidth=2)
        axes[1,0].set_title('训练 / 验证 损失走势', fontsize=13, fontweight='bold')
        axes[1,0].set_xlabel('epoch')
        axes[1,0].set_ylabel('loss')
        axes[1,0].legend()
        axes[1,0].grid(alpha=0.3)

    # 子图4：预测分布
    if pred_dist_df is not None and len(pred_dist_df) > 0:
        axes[1,1].bar(
            pred_dist_df['类别'],
            pred_dist_df['数量'],
            color=COLORS[:len(pred_dist_df)], edgecolor='black', alpha=0.85
        )
        for j, row in pred_dist_df.iterrows():
            axes[1,1].text(
                j,
                row['数量'] + 0.5,
                f"{row['占比(%)']:.1f}%",
                ha='center',
                fontsize=9
            )
        axes[1,1].set_title('测试集预测类别分布（推测现场占比）', fontsize=13, fontweight='bold')
        axes[1,1].set_ylabel('数量')
        axes[1,1].grid(axis='y', alpha=0.3)

    plt.suptitle('模型多维度表现仪表盘', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig('dashboard_overview.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # =======================
    # 1. 读取数据
    # =======================
    cwd = os.getcwd()
    history_df = safe_read_csv(os.path.join(cwd, 'training_history.csv'))
    class_df   = safe_read_csv(os.path.join(cwd, 'classification_results.csv'))
    pred_df    = safe_read_csv(os.path.join(cwd, 'test_predictions_details.csv'))

    # 预处理
    if history_df is not None:
        # 强制转数值类型
        for col in ['train_acc','val_acc','train_loss','val_loss']:
            history_df[col] = history_df[col].astype(float)

    if class_df is not None:
        class_df['精确率_float'] = class_df['精确率'].apply(percent_to_float)
        class_df['召回率_float'] = class_df['召回率'].apply(percent_to_float)
        class_df['F1_float']   = class_df['F1分数'].apply(percent_to_float)
        per_class_df = class_df[class_df['类别'] != '总体'].copy()
        overall_row  = class_df[class_df['类别'] == '总体'].copy()
    else:
        per_class_df = None
        overall_row = None

    # =======================
    # 2. 作图 & 关键数值分析
    # =======================
    # 2.1 每类别指标对比图
    draw_per_class_metrics(per_class_df)

    # 2.2 训练曲线 + 过拟合分析
    training_info = draw_train_val_curves(history_df)

    # 2.3 测试集预测分布
    pred_dist_df = draw_test_pred_distribution(pred_df)

    # 2.4 综合仪表盘
    build_dashboard_figure(per_class_df, history_df, pred_dist_df)

    # 2.5 风险标记（低召回/低精确/弱势类）
    risk_info = compute_risk_flags(per_class_df)

    # =======================
    # 3. 业务/部署向总结输出
    # =======================
    print("\n================ 模型多维度中文评估报告 ================\n")

    # --- 整体水平 ---
    print("【整体性能】")
    if overall_row is not None and len(overall_row) > 0:
        ov = overall_row.iloc[0]
        print(f"- 模型整体精确率(Precision): {ov['精确率_float']:.2f}%")
        print(f"- 模型整体召回率(Recall):   {ov['召回率_float']:.2f}%")
        print(f"- 模型整体F1分数:          {ov['F1_float']:.2f}%")
    else:
        print("- 未找到总体指标行（classification_results.csv 中可能缺少“总体”行）")

    # --- 训练稳定性 / 过拟合 ---
    print("\n【训练稳定性 / 过拟合风险】")
    if training_info is not None:
        print(f"- 最佳验证准确率出现在第 {training_info['best_epoch']} 轮: {training_info['best_val_acc']:.2f}%")
        print(f"- 当时训练准确率约为: {training_info['train_acc_at_best']:.2f}%")
        print(f"- 当时训练-验证准确率差距(过拟合gap): {training_info['gap_at_best']:.2f} 个百分点")
        print(f"- 最后一轮(epoch {training_info['final_epoch']}) 验证准确率约为: {training_info['final_val_acc']:.2f}%")
        print(f"- 最后一轮训练-验证差距: {training_info['final_gap']:.2f} 个百分点")
        print(f"- 最近几轮验证集loss平均变化(>0代表在升高): {training_info['val_loss_trend_last3']:.4f}")

        if training_info['final_gap'] > 5:
            print("  ▶ 警告：后期训练准确率明显高于验证准确率（gap>5个百分点），存在过拟合风险。")
        else:
            print("  ▶ 训练集与验证集表现较接近，过拟合可控。")
    else:
        print("- 无训练历史数据，无法判断过拟合。")

    # --- 类别层面分析 ---
    print("\n【按类别的可靠性】")
    if per_class_df is not None and len(per_class_df) > 0:
        weakest_cls = risk_info.get("weakest_class", None)
        weakest_f1  = risk_info.get("weakest_class_f1", None)
        if weakest_cls is not None:
            print(f"- 最薄弱类别：{weakest_cls}，其F1约为 {weakest_f1:.2f}%（需要重点关注）")

        if len(risk_info["low_recall"]) > 0:
            print(f"- 这些类别召回率<60%，模型容易漏检（实际使用中可能“看不到”它们）：{risk_info['low_recall']}")
        else:
            print("- 所有类别召回率均≥60% 或未触发漏检告警。")

        if len(risk_info["low_precision"]) > 0:
            print(f"- 这些类别精确率<60%，模型容易误报（会提高人工复核成本）：{risk_info['low_precision']}")
        else:
            print("- 所有类别精确率均≥60% 或未触发误报告警。")

        # 类别不平衡感知：如果某个类别F1远低于均值 => 数据不足或边界模糊
        mean_f1 = per_class_df['F1_float'].mean()
        bad_classes = per_class_df[per_class_df['F1_float'] < mean_f1 - 10]['类别'].tolist()
        if bad_classes:
            print(f"- 这些类别F1显著低于平均水平（平均F1约{mean_f1:.1f}%）：{bad_classes}，可能是样本量不足或特征区分度低。")
    else:
        print("- 未读取到按类别指标，无法分析类别可靠性。")

    # --- 输出倾向 / 现场分布 ---
    print("\n【推测现场输出分布 / 模型偏好】")
    if pred_dist_df is not None and len(pred_dist_df) > 0:
        top_row = pred_dist_df.iloc[0]
        print(f"- 模型在测试集上最常预测的类别：{top_row['类别']}，占比约 {top_row['占比(%)']:.2f}%")
        print("- 这可能意味着：在真实业务场景里，模型会更多地报出该类别的告警。")
        if len(pred_dist_df) > 1:
            tail_row = pred_dist_df.iloc[-1]
            print(f"- 模型最少预测的类别：{tail_row['类别']}，只占比 {tail_row['占比(%)']:.2f}%")
            print("  如果这类缺陷在真实环境中其实并不少见，那说明模型可能存在“看不到它/不敢报它”的风险。")
    else:
        print("- 未能读取测试集预测分布，无法判断模型的输出偏向。")

    # --- 部署建议 ---
    print("\n【部署建议 / 质检落地建议】")
    print("- 如果某些关键类别（比如严重安全隐患）召回率偏低：在上线阶段必须要求人工复核该类的低置信度样本，不能直接放行。")
    print("- 如果训练-验证 gap 偏大：建议引入更强的数据增强、类别均衡采样、或使用更早的最佳epoch权重而不是最终epoch。")
    print("- 如果某类精确率很低：实际落地时会产生大量误报，可以通过设置更高的置信度阈值来减少无效告警。")
    print("- 高价值方向：为最薄弱类别收集更多带框或高质量标注的样本，再单独做重采样/重加权训练。")

    print("\n================ 评估完成，图表已保存到当前目录 ================\n")
    print("已生成文件：")
    print(" - per_class_metrics.png               按类别精确率/召回率/F1对比")
    print(" - train_val_accuracy_curve_eval.png   训练/验证准确率曲线")
    print(" - train_val_loss_curve_eval.png       训练/验证损失曲线")
    print(" - train_val_gap.png                   过拟合风险曲线")
    print(" - test_pred_distribution.png          测试集输出类别分布")
    print(" - dashboard_overview.png              综合仪表盘（四宫格）")


if __name__ == "__main__":
    main()
