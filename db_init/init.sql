-- PostgreSQL 初始化脚本（docker-entrypoint-initdb.d）
-- 仅在数据卷首次初始化时执行。

-- 向量检索扩展，models/datasets.py 的 Vector 列依赖它
CREATE EXTENSION IF NOT EXISTS vector;

-- 通用工具扩展（生成 UUID 等）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 中文全文检索支持（可选，供后续数据集检索使用）
CREATE EXTENSION IF NOT EXISTS pg_trgm;
