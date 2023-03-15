from backend.helpers.images_helper import get_image, get_aligned_overlay
from backend.settings import overlay_data


def get_images(o_data=overlay_data):
    return get_image(o_data)


def get_aligned_images(transform, axis, value, session):
    session.append(transform, axis, value)
    aligned_overlay = get_aligned_overlay(overlay_data, session.transformations, session.initial_matrix)
    return get_images(aligned_overlay)
