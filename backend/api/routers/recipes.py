"""
Recipes Router
Handles recipe management, search, and bulk import.
"""

import logging
from uuid import UUID
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import get_current_active_user, pagination_params
from backend.api.schemas import (
    RecipeCreate,
    RecipeResponse,
    RecipeBulkImport,
    MessageResponse,
)
from backend.shared.database_v2 import get_session
from backend.shared.models_v2 import User, Recipe

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "",
    response_model=RecipeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create recipe",
    description="Add a new recipe to the database",
)
async def create_recipe(
    request: RecipeCreate,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> Recipe:
    """Create a new recipe."""
    recipe = Recipe(
        name=request.name,
        description=request.description,
        ingredients=request.ingredients,
        instructions=request.instructions,
        servings=request.servings,
        prep_time_minutes=request.prep_time_minutes,
        cook_time_minutes=request.cook_time_minutes,
        cuisine_type=request.cuisine_type,
        dietary_tags=request.dietary_tags,
        source_url=request.source_url,
    )
    
    session.add(recipe)
    await session.commit()
    await session.refresh(recipe)
    
    logger.info(f"Recipe created: {recipe.id} - {recipe.name}")
    
    return recipe


@router.post(
    "/import",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Bulk import recipes",
    description="Import multiple recipes at once (max 1000)",
)
async def bulk_import_recipes(
    request: RecipeBulkImport,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Bulk import recipes."""
    recipes = []
    
    for recipe_data in request.recipes:
        recipe = Recipe(
            name=recipe_data.name,
            description=recipe_data.description,
            ingredients=recipe_data.ingredients,
            instructions=recipe_data.instructions,
            servings=recipe_data.servings,
            prep_time_minutes=recipe_data.prep_time_minutes,
            cook_time_minutes=recipe_data.cook_time_minutes,
            cuisine_type=recipe_data.cuisine_type,
            dietary_tags=recipe_data.dietary_tags,
            source_url=recipe_data.source_url,
        )
        recipes.append(recipe)
    
    session.add_all(recipes)
    await session.commit()
    
    logger.info(f"Bulk imported {len(recipes)} recipes")
    
    return MessageResponse(
        message=f"Successfully imported {len(recipes)} recipes",
        success=True,
    )


@router.get(
    "/search",
    response_model=List[RecipeResponse],
    summary="Search recipes",
    description="Search recipes by ingredients and dietary preferences",
)
async def search_recipes(
    ingredients: str | None = Query(None, description="Comma-separated ingredients"),
    diet: str | None = Query(None, description="Dietary preference (vegetarian, vegan, etc.)"),
    cuisine: str | None = Query(None, description="Cuisine type"),
    max_prep_time: int | None = Query(None, ge=0, description="Maximum prep time in minutes"),
    session: AsyncSession = Depends(get_session),
    pagination: dict = Depends(pagination_params),
) -> List[Recipe]:
    """Search recipes with filters."""
    query = select(Recipe).where(Recipe.deleted_at.is_(None))
    
    # Filter by ingredients
    if ingredients:
        ingredient_list = [i.strip() for i in ingredients.split(",")]
        for ingredient in ingredient_list:
            query = query.where(
                Recipe.ingredients.any(ingredient.lower())
            )
    
    # Filter by dietary tags
    if diet:
        query = query.where(
            Recipe.dietary_tags.contains([diet.lower()])
        )
    
    # Filter by cuisine
    if cuisine:
        query = query.where(Recipe.cuisine_type.ilike(f"%{cuisine}%"))
    
    # Filter by prep time
    if max_prep_time is not None:
        query = query.where(Recipe.prep_time_minutes <= max_prep_time)
    
    # Apply pagination
    query = query.offset(pagination["skip"]).limit(pagination["limit"])
    
    result = await session.execute(query)
    recipes = result.scalars().all()
    
    return list(recipes)


@router.get(
    "/{recipe_id}",
    response_model=RecipeResponse,
    summary="Get recipe",
    description="Get recipe details by ID",
)
async def get_recipe(
    recipe_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> Recipe:
    """Get recipe by ID."""
    result = await session.execute(
        select(Recipe).where(
            Recipe.id == recipe_id,
            Recipe.deleted_at.is_(None),
        )
    )
    
    recipe = result.scalar_one_or_none()
    
    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recipe not found",
        )
    
    return recipe


@router.get(
    "",
    response_model=List[RecipeResponse],
    summary="List recipes",
    description="Get all recipes with pagination",
)
async def list_recipes(
    session: AsyncSession = Depends(get_session),
    pagination: dict = Depends(pagination_params),
) -> List[Recipe]:
    """List all recipes."""
    result = await session.execute(
        select(Recipe)
        .where(Recipe.deleted_at.is_(None))
        .offset(pagination["skip"])
        .limit(pagination["limit"])
        .order_by(Recipe.created_at.desc())
    )
    
    recipes = result.scalars().all()
    
    return list(recipes)
