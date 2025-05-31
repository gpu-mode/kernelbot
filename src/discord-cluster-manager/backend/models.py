from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Leaderboard(Base):
    __tablename__ = "leaderboard"
    __table_args__ = {"schema": "leaderboard"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, unique=True, nullable=False)
    deadline = Column(DateTime(timezone=True), nullable=False)
    task = Column(Text, nullable=False)  # JSON string representation of LeaderboardTask
    creator_id = Column(BigInteger, nullable=False, default=-1)
    forum_id = Column(BigInteger, nullable=False, default=-1)
    secret_seed = Column(BigInteger, nullable=False, default=func.floor(func.random() * 2147483648))

    # Relationships
    gpu_types = relationship("GpuType", back_populates="leaderboard", cascade="all, delete-orphan")
    submissions = relationship(
        "Submission", back_populates="leaderboard", cascade="all, delete-orphan"
    )


class GpuType(Base):
    __tablename__ = "gpu_type"
    __table_args__ = {"schema": "leaderboard"}

    leaderboard_id = Column(Integer, ForeignKey("leaderboard.leaderboard.id"), primary_key=True)
    gpu_type = Column(Text, primary_key=True)

    # Relationships
    leaderboard = relationship("Leaderboard", back_populates="gpu_types")


class UserInfo(Base):
    __tablename__ = "user_info"
    __table_args__ = {"schema": "leaderboard"}

    id = Column(Text, primary_key=True)
    user_name = Column(Text)
    cli_id = Column(String(255), default=None)
    cli_valid = Column(Boolean, default=False)
    cli_auth_provider = Column(String(255), default=None)
    created_at = Column(DateTime(timezone=True), default=func.now())

    # Relationships
    submissions = relationship("Submission", back_populates="user")


class CodeFile(Base):
    __tablename__ = "code_files"
    __table_args__ = {"schema": "leaderboard"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(Text, nullable=False)
    hash = Column(Text)  # Generated column in PostgreSQL

    # Relationships
    submissions = relationship("Submission", back_populates="code_file")


class Submission(Base):
    __tablename__ = "submission"
    __table_args__ = {"schema": "leaderboard"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    leaderboard_id = Column(Integer, ForeignKey("leaderboard.leaderboard.id"), nullable=False)
    file_name = Column(Text, nullable=False)
    user_id = Column(Text, ForeignKey("leaderboard.user_info.id"), nullable=False)
    code_id = Column(Integer, ForeignKey("leaderboard.code_files.id"), nullable=False)
    submission_time = Column(DateTime(timezone=True), nullable=False)
    done = Column(Boolean, default=False)

    # Relationships
    leaderboard = relationship("Leaderboard", back_populates="submissions")
    user = relationship("UserInfo", back_populates="submissions")
    code_file = relationship("CodeFile", back_populates="submissions")
    runs = relationship("Run", back_populates="submission", cascade="all, delete-orphan")


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = {"schema": "leaderboard"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    submission_id = Column(Integer, ForeignKey("leaderboard.submission.id"), nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=False)
    mode = Column(Text, nullable=False)
    secret = Column(Boolean, nullable=False)
    runner = Column(Text, nullable=False)
    score = Column(Numeric)
    passed = Column(Boolean, nullable=False)
    compilation = Column(JSON)
    meta = Column(JSON)
    result = Column(JSON)
    system_info = Column(JSON, nullable=False)

    # Relationships
    submission = relationship("Submission", back_populates="runs")
