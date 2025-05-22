import pymongo
from pymongo import MongoClient
from typing import List, Dict, Any, Optional
from bson.objectid import ObjectId
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageRepository:
    def __init__(self, mongo_uri=None):
        mongo_uri = mongo_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.client = MongoClient(mongo_uri)

        db_name = os.getenv("MONGODB_DB", "chat-app")
        self.db = self.client[db_name]

        self.messages = self.db["Messages"]
        self.text_messages = self.db["TextMessages"]
        self.conservations = self.db["Conservations"]
        self.users = self.db["Users"]

        logger.info(f"Connected to MongoDB: {mongo_uri}, database: {db_name}")

    def get_conservation_messages(
        self, conservation_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        try:
            if not ObjectId.is_valid(conservation_id):
                logger.error(f"Invalid conservation ID: {conservation_id}")
                return []

            conservation_oid = ObjectId(conservation_id)

            messages = list(
                self.messages.find(
                    {"conservation": conservation_oid, "isDeleted": {"$ne": True}},
                    {"__v": 0},
                )
                .sort("createdAt", -1)
                .limit(limit)
            )

            messages.reverse()

            for message in messages:
                if message["type"] == "text":
                    text_message = self.text_messages.find_one({"_id": message["_id"]})
                    if text_message:
                        message["content"] = {"text": text_message.get("text", "")}

                message["_id"] = str(message["_id"])

                if isinstance(message.get("sender"), ObjectId):
                    message["sender"] = str(message["sender"])

                if isinstance(message.get("conservation"), ObjectId):
                    message["conservation"] = str(message["conservation"])

            logger.info(
                f"Retrieved {len(messages)} messages for conservation {conservation_id}"
            )
            return messages

        except Exception as e:
            logger.error(f"Error retrieving messages: {str(e)}")
            return []

    def get_conservation_members(self, conservation_id: str) -> List[Dict[str, Any]]:
        try:
            if not ObjectId.is_valid(conservation_id):
                return []

            conservation = self.conservations.find_one(
                {"_id": ObjectId(conservation_id)}
            )
            if not conservation or "members" not in conservation:
                return []

            members = []
            for member in conservation["members"]:
                if "user" in member and isinstance(member["user"], ObjectId):
                    user = self.users.find_one({"_id": member["user"]})
                    if user:
                        members.append(
                            {
                                "user_id": str(user["_id"]),
                                "userName": user.get("userName", "User"),
                                "email": user.get("email", ""),
                                "nickname": member.get("nickname", ""),
                                "role": member.get("role", "MEMBER"),
                            }
                        )

            return members

        except Exception as e:
            logger.error(f"Error retrieving conservation members: {str(e)}")
            return []

    def get_conservation_info(self, conservation_id: str) -> Optional[Dict[str, Any]]:
        try:
            if not ObjectId.is_valid(conservation_id):
                return None

            conservation = self.conservations.find_one(
                {"_id": ObjectId(conservation_id)}
            )
            if not conservation:
                return None

            return {
                "id": str(conservation["_id"]),
                "type": conservation.get("type", "GROUP"),
                "member_count": len(conservation.get("members", [])),
                "attributes": conservation.get("conservationAttributes", {}),
            }

        except Exception as e:
            logger.error(f"Error retrieving conservation info: {str(e)}")
            return None


_message_repository = None


def get_message_repository():
    global _message_repository
    if _message_repository is None:
        _message_repository = MessageRepository()
    return _message_repository
