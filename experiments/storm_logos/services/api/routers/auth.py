"""Auth Router: User registration and authentication."""

from fastapi import APIRouter, HTTPException, status, Depends

from ..models import UserCreate, UserLogin, UserResponse, TokenResponse
from ..deps import get_user_graph, create_token, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def register(data: UserCreate):
    """Register a new user."""
    ug = get_user_graph()

    # Check if username exists
    existing = ug.get_user(data.username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )

    # Create user
    user = ug.create_user(data.username, data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create user"
        )

    # Create token
    token = create_token(user.user_id, user.username)

    return TokenResponse(
        access_token=token,
        user=UserResponse(
            user_id=user.user_id,
            username=user.username,
            created_at=user.created_at,
        )
    )


@router.post("/login", response_model=TokenResponse)
async def login(data: UserLogin):
    """Login and get access token."""
    ug = get_user_graph()

    user = ug.authenticate(data.username, data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    token = create_token(user.user_id, user.username)

    return TokenResponse(
        access_token=token,
        user=UserResponse(
            user_id=user.user_id,
            username=user.username,
            created_at=user.created_at,
        )
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user info."""
    ug = get_user_graph()
    user = ug.get_user(current_user["username"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        created_at=user.created_at,
    )
