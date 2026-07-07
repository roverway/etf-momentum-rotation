# =====================================================================================
# 1. 导入我们平台支持的第三方python模块
# =====================================================================================
import pandas as pd
import copy

# =====================================================================================
# 2. 策略配置区
# =====================================================================================
ETF_POOL = [
    '513100.XSHG',  # 纳指ETF (2013年上市)
    '159915.XSHE',  # 创业板ETF (2011年上市)
    '518880.XSHG',  # 黄金ETF
    # '511010.XSHG',  # 国债ETF
    # '588000.XSHG',  # 科创50ETF
    # '563300.XSHG',  # 中证2000ETF
    # '513180.XSHG',  # 恒生科技指数ETF
    '512890.XSHG',  # 红利低波ETF
    # '510050.XSHG',  # 50ETF
    # '162411.XSHE',  # 华宝油气 (2012年上市)
]
CHECK_RANGE = 22   # 指交易日，22个交易日模拟1个月

# =====================================================================================
# 3. [已修改] 核心逻辑函数
# =====================================================================================

def calculate_momentum_signal(context, base_date):
    """
    根据给定的基准日期(base_date)，计算出下一交易日的目标ETF。
    """
    end_date = get_previous_trading_date(base_date)
    start_date = get_previous_trading_date(end_date, n=context.check_range - 1)

    all_prices = get_price(
        context.etf_codes, start_date=start_date, end_date=end_date,
        frequency='1d', fields='close', adjust_type='pre'
    )

    valid_returns = {}
    if all_prices is None or all_prices.empty:
        return None

    if isinstance(all_prices, pd.DataFrame):
        if all_prices.shape[0] < context.check_range: return None
        start_prices, last_prices = all_prices.iloc[0], all_prices.iloc[-1]
        etf_returns = (last_prices - start_prices) / start_prices
        valid_returns = etf_returns.dropna().to_dict()
    elif isinstance(all_prices, pd.Series):
        if len(all_prices) < context.check_range: return None
        the_one_etf = context.etf_codes[0]
        start_price, last_price = all_prices.iloc[0], all_prices.iloc[-1]
        if pd.notna(start_price) and pd.notna(last_price):
            valid_returns[the_one_etf] = (last_price - start_price) / start_price

    if not valid_returns:
        return None

    best_etf = max(valid_returns, key=valid_returns.get)
    if valid_returns[best_etf] > 0:
        return best_etf
    else:
        return None

def log_trade_signal(current_position, next_day_target):
    """
    一个独立的信号播报函数，根据当前持仓和次日目标，打印出明确的中文操作指令。
    """
    # [关键修正] 直接使用品种代码，不再试图获取中文名，以保证最大兼容性
    current_name = current_position or "空仓"
    target_name = next_day_target or "空仓"

    logger.info(f"  > 当前实盘持仓: 【{current_name}】")
    logger.info(f"  > 次日交易目标: 【{target_name}】")
    logger.info("-" * 40)

    action = ""
    if current_position == next_day_target:
        action = f"✅ 操作建议: 保持现有持仓【{current_name}】不变。(HOLD)"
    elif current_position is None and next_day_target is not None:
        action = f"🚀 操作建议: 从【空仓】买入【{target_name}】。(BUY)"
    elif current_position is not None and next_day_target is None:
        action = f"🛑 操作建议: 卖出【{current_name}】，清仓观望。(SELL TO CASH)"
    else:
        action = f"🔁 操作建议: 卖出【{current_name}】，换仓买入【{target_name}】。(SWITCH)"

    logger.info(action)


# =====================================================================================
# 4. 初始化函数 (init)
# =====================================================================================
def init(context):
    context.etf_codes = ETF_POOL
    context.check_range = CHECK_RANGE
    context.target_etf = None

    context.pnl_by_instrument = {}
    context.trade_records = {}
    context.pending_pnl_calculation = None

    logger.info("策略已启动，已启用【最终完美版v13】。")
    if hasattr(context, 'run_info'):
        logger.info("RunInfo: {}".format(context.run_info))

# =====================================================================================
# 5. 盘前处理函数 (before_trading)
# =====================================================================================
def before_trading(context):
    if context.pending_pnl_calculation:
        pending_info = context.pending_pnl_calculation
        instrument_to_calc = pending_info['instrument']
        yesterday = get_previous_trading_date(context.now)

        prices = get_price(instrument_to_calc, start_date=yesterday, end_date=yesterday, frequency='1d', fields='close', adjust_type='none')

        if prices is not None and not prices.empty:
            exit_price = prices.iloc[0]
            realized_pnl = (exit_price - pending_info['entry_price']) * pending_info['quantity']
            context.pnl_by_instrument[instrument_to_calc] = context.pnl_by_instrument.get(instrument_to_calc, 0) + realized_pnl
        else:
            logger.error(f"无法获取 {instrument_to_calc} 在 {yesterday.date()} 的价格，该笔平仓盈亏无法计算。")
        context.pending_pnl_calculation = None

    context.target_etf = calculate_momentum_signal(context, context.now)

# =====================================================================================
# 6. 交易执行函数 (handle_bar)
# =====================================================================================
def handle_bar(context, bar_dict):
    current_holding = next(iter(context.portfolio.positions), None)
    target_etf = context.target_etf
    if target_etf == current_holding:
        return

    if context.portfolio.positions:
        for stock_code in list(context.portfolio.positions.keys()):
            if stock_code != target_etf:
                order_target_percent(stock_code, 0)
    if target_etf:
        order_target_percent(target_etf, 1)

# =====================================================================================
# 7. 盘后处理函数 (after_trading)
# =====================================================================================
def after_trading(context):
    for instrument in list(context.trade_records.keys()):
        if instrument != context.target_etf:
            context.pending_pnl_calculation = {
                'instrument': instrument,
                'entry_price': context.trade_records[instrument]['avg_price'],
                'quantity': context.trade_records[instrument]['quantity']
            }
            del context.trade_records[instrument]

    for instrument, position in context.portfolio.positions.items():
        if instrument not in context.trade_records and position.quantity > 0:
            context.trade_records[instrument] = {
                'avg_price': position.avg_price,
                'quantity': position.quantity
            }

    if hasattr(context, 'run_info') and context.now.date() == context.run_info.end_date:

        if context.pending_pnl_calculation:
            pending_info = context.pending_pnl_calculation
            instrument_to_calc = pending_info['instrument']
            last_day = context.now.date()
            prices = get_price(instrument_to_calc, start_date=last_day, end_date=last_day, frequency='1d', fields='close', adjust_type='none')
            if prices is not None and not prices.empty:
                exit_price = prices.iloc[0]
                realized_pnl = (exit_price - pending_info['entry_price']) * pending_info['quantity']
                context.pnl_by_instrument[instrument_to_calc] = context.pnl_by_instrument.get(instrument_to_calc, 0) + realized_pnl
            context.pending_pnl_calculation = None

        # [关键修正] 使用今天的日期作为基准来计算次日信号
        next_day_target = calculate_momentum_signal(context, context.now)
        current_position = next(iter(context.portfolio.positions), None)

        logger.info("\n" + "=" * 20 + " 明日实盘操作指令 " + "=" * 20)
        next_trade_date = get_next_trading_date(context.now)
        logger.info(f"指令生成于: {context.now.date()} 收盘后")
        logger.info(f"适用于: 【{next_trade_date.date()}】开盘操作")

        log_trade_signal(current_position, next_day_target)
        logger.info("=" * 60)

        logger.info("\n" + "=" * 60)
        logger.info("【回测全周期分品种收益贡献汇总】")
        final_pnl_contribution = context.pnl_by_instrument.copy()
        for inst, pos in context.portfolio.positions.items():
            final_pnl_contribution[inst] = final_pnl_contribution.get(inst, 0) + pos.pnl

        logger.info("-" * 60)
        for etf in ETF_POOL:
            if etf not in final_pnl_contribution: final_pnl_contribution[etf] = 0.0

        sorted_pnl = sorted(final_pnl_contribution.items(), key=lambda item: item[1], reverse=True)
        for instrument, pnl in sorted_pnl:
             logger.info(f"  - 品种: {instrument:<15} 总贡献: {pnl:,.2f}")

        logger.info("-" * 60)
        total_contribution = sum(final_pnl_contribution.values())
        logger.info(f"所有品种贡献合计(毛利): {total_contribution:,.2f}")
        if hasattr(context.portfolio, 'pnl'):
            logger.info(f"策略组合最终总盈亏 (净利): {context.portfolio.pnl:,.2f}")
        logger.info("="*60 + "\n")
